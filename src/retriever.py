"""Graph retriever: question -> linked entities + intents -> ranked graph facts.

Retrieval is a three-step process:

1. Entity linking   - movie titles and person names from the graph catalogue are
                      matched against the question (exact phrase first, then a
                      unique-surname match, then a fuzzy match for typos).
2. Intent detection - keyword rules decide *which* relationship types the user
                      is asking about (DIRECTED, ACTED_IN, WROTE, ...), plus an
                      optional release-year constraint.
3. Graph retrieval  - for every linked entity the 1-hop neighbourhood is fetched
                      from the graph store; facts that match the detected intent
                      are ranked first and the list is cut to ``max_facts``.
                      If nothing is linked, a TF-IDF search over one text
                      document per movie (title + tagline + people) is used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENT_RULES = {
    "DIRECTED": r"\bdirect(ed|or|ors|s|ing)?\b|\bfilmmaker",
    "ACTED_IN": r"\bcast\b|\bstar(s|red|ring)?\b|\bact(ed|or|ors|ress|ing)?\b|"
                r"\bwho (was|were|is|are) in\b|\bappear",
    "ROLE":     r"\brole|\bplay(ed|s)?\b|\bcharacter",
    "RELEASED": r"\breleas|\bwhat year\b|\bwhen (was|did|is)\b|\bwhich year\b",
    "TAGLINE":  r"\btagline|\bslogan|\bwhat is .* about\b|\bplot\b",
    "WROTE":    r"\bwr(ote|ite|iter|itten)\b|\bscreenplay|\bscript",
    "PRODUCED": r"\bproduc",
    "REVIEWED": r"\breview|\brating|\brated\b|\bcritic|\bscore",
    "BORN":     r"\bborn\b|\bbirth|\bhow old\b|\bage\b",
}
REL_FOR_INTENT = {"ROLE": "ACTED_IN"}
SURNAME_STOP = {"the", "and", "jr", "jr."}


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' /]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    detail: str = ""
    score: float = 1.0

    def as_text(self) -> str:
        d = f" [{self.detail}]" if self.detail else ""
        return f"({self.subject}) -[{self.relation}]-> ({self.obj}){d}"


@dataclass
class Retrieval:
    question: str
    movies: list = field(default_factory=list)
    people: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    years: tuple | None = None
    strategy: str = ""
    facts: list = field(default_factory=list)
    queries: list = field(default_factory=list)   # Cypher templates used

    def context(self) -> str:
        return "\n".join(f"{i+1}. {f.as_text()}" for i, f in enumerate(self.facts))


class GraphRetriever:
    def __init__(self, store, max_facts: int = 25, fuzzy_threshold: int = 88):
        self.store = store
        self.max_facts = max_facts
        self.fuzzy_threshold = fuzzy_threshold
        self.titles = store.movie_titles()
        self.people = store.person_names()
        # surname -> full name, kept only when the surname is unique
        surnames = {}
        for p in self.people:
            last = _norm(p).split()[-1]
            if last not in SURNAME_STOP and len(last) > 3:
                surnames.setdefault(last, []).append(p)
        self.surnames = {k: v[0] for k, v in surnames.items() if len(v) == 1}
        # TF-IDF index for the no-entity fallback
        self.docs = store.all_movie_docs()
        texts = [f"{d['title']} {d['title']} {d.get('tagline') or ''} "
                 f"{' '.join(d['people'])}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------ 1
    def link_entities(self, question: str):
        q = f" {_norm(question)} "
        movies, people = [], []
        # exact phrase match, longest names first so "The Matrix Reloaded"
        # wins over "The Matrix"
        candidates = [(t, "movie") for t in self.titles] + \
                     [(p, "person") for p in self.people]
        for name, kind in sorted(candidates, key=lambda c: -len(c[0])):
            n = f" {_norm(name)} "
            if n in q:
                (movies if kind == "movie" else people).append(name)
                q = q.replace(n, " # ")
        # unique surname ("Hanks" -> Tom Hanks)
        for tok in q.split():
            if tok in self.surnames and self.surnames[tok] not in people:
                people.append(self.surnames[tok])
                q = q.replace(f" {tok} ", " # ")
        # fuzzy match on word windows for typos ("Keanu Reaves")
        words = q.split()
        for name, kind in candidates:
            target = (movies if kind == "movie" else people)
            if name in target:
                continue
            n = _norm(name)
            size = len(n.split())
            if size < 2:        # single-word names are too risky to fuzz
                continue
            for i in range(len(words) - size + 1):
                window = " ".join(words[i:i + size])
                if "#" in window:
                    continue
                if fuzz.ratio(window, n) >= self.fuzzy_threshold:
                    target.append(name)
                    break
        return movies, people

    # ------------------------------------------------------------------ 2
    @staticmethod
    def detect_intents(question: str):
        q = question.lower()
        intents = [k for k, rx in INTENT_RULES.items() if re.search(rx, q)]
        years = None
        m = re.search(r"between (\d{4}) and (\d{4})", q)
        if m:
            years = (int(m.group(1)), int(m.group(2)))
        elif (m := re.search(r"\b(19|20)(\d)0s\b", q)):
            start = int(m.group(1) + m.group(2) + "0")
            years = (start, start + 9)
        elif (m := re.search(r"\b(after|since|before|in)\s+(\d{4})\b", q)):
            y = int(m.group(2))
            years = {"after": (y + 1, 2100), "since": (y, 2100),
                     "before": (1900, y - 1), "in": (y, y)}[m.group(1)]
        return intents, years

    # ------------------------------------------------------------------ 3
    def retrieve(self, question: str) -> Retrieval:
        r = Retrieval(question=question)
        r.movies, r.people = self.link_entities(question)
        r.intents, r.years = self.detect_intents(question)
        wanted = {REL_FOR_INTENT.get(i, i) for i in r.intents}
        facts: list[Fact] = []

        for title in r.movies:
            r.queries.append("movie_profile")
            prof = self.store.movie_profile(title)
            label = f"{title} ({prof['released']})"
            facts.append(Fact(label, "RELEASED", str(prof["released"]),
                              score=3 if "RELEASED" in wanted else 1))
            if prof.get("tagline"):
                facts.append(Fact(label, "TAGLINE", prof["tagline"],
                                  score=3 if "TAGLINE" in wanted else 1))
            for l in prof["links"]:
                detail = self._detail(l)
                s = 1 + 2 * (l["rel"] in wanted) + 2 * (l["person"] in r.people)
                facts.append(Fact(l["person"], l["rel"], label, detail, s))

        for name in r.people:
            r.queries.append("person_profile")
            prof = self.store.person_profile(name)
            if prof.get("born"):
                facts.append(Fact(name, "BORN", str(prof["born"]),
                                  score=3 if "BORN" in wanted else 1))
            for l in prof["links"]:
                if r.years and not (r.years[0] <= (l["released"] or 0) <= r.years[1]):
                    continue
                movie = f"{l['movie']} ({l['released']})"
                s = 1 + 2 * (l["rel"] in wanted) + 2 * (l["movie"] in r.movies)
                facts.append(Fact(name, l["rel"], movie, self._detail(l), s))

        if len(r.people) >= 2:
            r.queries.append("shared_movies")
            a, b = r.people[:2]
            for row in self.store.shared_movies(a, b):
                facts.append(Fact(f"{a} & {b}", "BOTH_WORKED_ON",
                                  f"{row['movie']} ({row['released']})",
                                  f"{a}: {row['rel_a']}, {b}: {row['rel_b']}", 6))

        if r.years and not r.people and not r.movies:
            r.queries.append("movies_by_year")
            for row in self.store.movies_by_year(*r.years):
                facts.append(Fact(row["title"], "RELEASED", str(row["released"]),
                                  score=3))

        if facts:
            r.strategy = "graph (entity-linked)" if (r.movies or r.people) \
                else "graph (year filter)"
        else:
            r.strategy = "tf-idf fallback"
            r.queries.append("all_movie_docs")
            facts = self._keyword_fallback(question)

        # de-duplicate (a fact can be reached from both the movie and the person)
        uniq = {}
        for f in facts:
            key = (f.subject.split(" (")[0], f.relation, f.obj.split(" (")[0])
            if key not in uniq or f.score > uniq[key].score:
                uniq[key] = f
        r.facts = sorted(uniq.values(), key=lambda f: -f.score)[: self.max_facts]
        return r

    @staticmethod
    def _detail(link):
        if link.get("roles"):
            return "as " + ", ".join(link["roles"])
        if link.get("summary"):
            return f"rating {link['rating']}: \"{link['summary']}\""
        return ""

    def _keyword_fallback(self, question, k=3, min_score=0.08):
        sims = cosine_similarity(self.vectorizer.transform([question]),
                                 self.doc_matrix)[0]
        facts = []
        for idx in sims.argsort()[::-1][:k]:
            if sims[idx] < min_score:
                break
            d = self.docs[idx]
            label = f"{d['title']} ({d['released']})"
            facts.append(Fact(label, "TAGLINE", d.get("tagline") or "(none)",
                              f"similarity {sims[idx]:.2f}", float(sims[idx]) + 1))
            facts.append(Fact(label, "PEOPLE", ", ".join(d["people"][:6]),
                              score=float(sims[idx])))
        return facts
