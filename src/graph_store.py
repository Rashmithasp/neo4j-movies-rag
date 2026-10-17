"""Graph access layer for the Neo4j Movies dataset.

Two interchangeable back ends expose the same retrieval methods:

* Neo4jStore  - talks to a running Neo4j server through the official Python
                driver and answers every call with a parameterised Cypher query.
* LocalStore  - loads the very same ``movies.cypher`` script into an in-memory
                graph. Used for unit tests / offline demos when no Neo4j server
                is available. Each method mirrors the Cypher of the Neo4j
                back end so both return records of the same shape.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "movies.cypher"

# ---------------------------------------------------------------------------
# Cypher templates (used by Neo4jStore, shown in the UI for transparency)
# ---------------------------------------------------------------------------
CYPHER = {
    "catalogue_movies": "MATCH (m:Movie) RETURN m.title AS title ORDER BY title",
    "catalogue_people": "MATCH (p:Person) RETURN p.name AS name ORDER BY name",
    "movie_profile": """
MATCH (m:Movie {title: $title})
OPTIONAL MATCH (p:Person)-[r]->(m)
RETURN m.title AS title, m.released AS released, m.tagline AS tagline,
       collect({person: p.name, rel: type(r), roles: r.roles,
                summary: r.summary, rating: r.rating}) AS links""",
    "person_profile": """
MATCH (p:Person {name: $name})
OPTIONAL MATCH (p)-[r]->(m:Movie)
RETURN p.name AS name, p.born AS born,
       collect({movie: m.title, released: m.released, rel: type(r),
                roles: r.roles, summary: r.summary, rating: r.rating}) AS links""",
    "shared_movies": """
MATCH (a:Person {name: $a})-[ra]->(m:Movie)<-[rb]-(b:Person {name: $b})
RETURN m.title AS movie, m.released AS released,
       type(ra) AS rel_a, type(rb) AS rel_b""",
    "movies_by_year": """
MATCH (m:Movie) WHERE m.released >= $start AND m.released <= $end
RETURN m.title AS title, m.released AS released ORDER BY released, title""",
    "all_movie_docs": """
MATCH (m:Movie)
OPTIONAL MATCH (p:Person)-[r]->(m)
RETURN m.title AS title, m.released AS released, m.tagline AS tagline,
       collect(p.name) AS people""",
    "stats": """
MATCH (n) WITH count(n) AS nodes
MATCH ()-[r]->() RETURN nodes, count(r) AS rels""",
}


# ---------------------------------------------------------------------------
# Neo4j back end
# ---------------------------------------------------------------------------
class Neo4jStore:
    name = "Neo4j"

    def __init__(self, uri=None, user=None, password=None, database=None):
        from neo4j import GraphDatabase

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(user or os.getenv("NEO4J_USER", "neo4j"),
                  password or os.getenv("NEO4J_PASSWORD", "password")),
        )
        self.driver.verify_connectivity()

    def _run(self, key, **params):
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(CYPHER[key], **params)]

    # catalogue -------------------------------------------------------------
    def movie_titles(self):
        return [r["title"] for r in self._run("catalogue_movies")]

    def person_names(self):
        return [r["name"] for r in self._run("catalogue_people")]

    # retrieval -------------------------------------------------------------
    def movie_profile(self, title):
        rows = self._run("movie_profile", title=title)
        if not rows:
            return None
        row = rows[0]
        row["links"] = [l for l in row["links"] if l["person"]]
        return row

    def person_profile(self, name):
        rows = self._run("person_profile", name=name)
        if not rows:
            return None
        row = rows[0]
        row["links"] = [l for l in row["links"] if l["movie"]]
        return row

    def shared_movies(self, a, b):
        return self._run("shared_movies", a=a, b=b)

    def movies_by_year(self, start, end):
        return self._run("movies_by_year", start=start, end=end)

    def all_movie_docs(self):
        return self._run("all_movie_docs")

    def stats(self):
        return self._run("stats")[0]


# ---------------------------------------------------------------------------
# Offline back end: parses movies.cypher into an in-memory property graph
# ---------------------------------------------------------------------------
_VALUE = r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|\[[^\]]*\]"""
_MAP = r"\{((?:" + _VALUE + r"|[^{}'\"])*)\}"
_NODE = re.compile(r"(MERGE|MATCH)\s*\((\w+):(\w+)\s*" + _MAP + r"\)"
                   r"(?:[ \t]*ON CREATE SET[ \t]*(.*?))?(?=\n|$)")
_REL = re.compile(r"MERGE\s*\((\w+)\)-\[:(\w+)\s*(?:" + _MAP + r")?\]->\((\w+)\)")
_PAIR = re.compile(r"(\w+)\s*:\s*(" + _VALUE + r")")
_SET = re.compile(r"\w+\.(\w+)\s*=\s*(" + _VALUE + r")")


def _lit(v):
    v = v.strip()
    if v[0] in "'\"":
        return v[1:-1].replace("\\'", "'").replace('\\"', '"')
    if v.startswith("["):
        return [_lit(x) for x in re.findall(_VALUE, v[1:-1])]
    return float(v) if "." in v else int(v)


class LocalStore:
    name = "Local mirror"

    def __init__(self, path=DATA_FILE):
        self.nodes = {}                  # key -> {"label":..., props}
        self.rels = []                   # (src_key, type, dst_key, props)
        self._load(Path(path).read_text(encoding="utf-8"))
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        for s, t, d, p in self.rels:
            self.out[s].append((t, d, p))
            self.inc[d].append((t, s, p))

    def _load(self, script):
        seen = set()
        for block in script.split(";"):
            env = {}
            for m in _NODE.finditer(block):
                _, var, label, body, sets = m.groups()
                pairs = _PAIR.findall(body)
                key, val = pairs[0]
                nid = (label, _lit(val))
                node = self.nodes.setdefault(nid, {"label": label, key: _lit(val)})
                for k, v in pairs[1:]:
                    node.setdefault(k, _lit(v))
                if sets:
                    for k, v in _SET.findall(sets):
                        node.setdefault(k, _lit(v))
                env[var] = nid
            for m in _REL.finditer(block):
                a, typ, props, b = m.groups()
                key = (env[a], typ, env[b])
                if key in seen:          # MERGE semantics: no duplicates
                    continue
                seen.add(key)
                pd = {k: _lit(v) for k, v in _PAIR.findall(props or "")}
                self.rels.append((env[a], typ, env[b], pd))

    # catalogue -------------------------------------------------------------
    def movie_titles(self):
        return sorted(k[1] for k in self.nodes if k[0] == "Movie")

    def person_names(self):
        return sorted(k[1] for k in self.nodes if k[0] == "Person")

    # retrieval (mirrors CYPHER[...]) --------------------------------------
    def movie_profile(self, title):
        key = ("Movie", title)
        if key not in self.nodes:
            return None
        m = self.nodes[key]
        links = [{"person": s[1], "rel": t, "roles": p.get("roles"),
                  "summary": p.get("summary"), "rating": p.get("rating")}
                 for t, s, p in self.inc[key] if s[0] == "Person"]
        return {"title": title, "released": m.get("released"),
                "tagline": m.get("tagline"), "links": links}

    def person_profile(self, name):
        key = ("Person", name)
        if key not in self.nodes:
            return None
        links = [{"movie": d[1], "released": self.nodes[d].get("released"),
                  "rel": t, "roles": p.get("roles"), "summary": p.get("summary"),
                  "rating": p.get("rating")}
                 for t, d, p in self.out[key] if d[0] == "Movie"]
        return {"name": name, "born": self.nodes[key].get("born"), "links": links}

    def shared_movies(self, a, b):
        ka, kb = ("Person", a), ("Person", b)
        b_links = {(d): t for t, d, _ in self.out[kb] if d[0] == "Movie"}
        rows = []
        for t, d, _ in self.out[ka]:
            if d in b_links:
                rows.append({"movie": d[1], "released": self.nodes[d].get("released"),
                             "rel_a": t, "rel_b": b_links[d]})
        return rows

    def movies_by_year(self, start, end):
        rows = [{"title": k[1], "released": n.get("released")}
                for k, n in self.nodes.items()
                if k[0] == "Movie" and start <= (n.get("released") or 0) <= end]
        return sorted(rows, key=lambda r: (r["released"], r["title"]))

    def all_movie_docs(self):
        out = []
        for k, n in self.nodes.items():
            if k[0] == "Movie":
                out.append({"title": k[1], "released": n.get("released"),
                            "tagline": n.get("tagline"),
                            "people": [s[1] for _, s, _ in self.inc[k]]})
        return out

    def stats(self):
        return {"nodes": len(self.nodes), "rels": len(self.rels)}


def get_store(prefer="auto"):
    """Return a Neo4jStore if reachable (or requested), else the LocalStore."""
    if prefer in ("auto", "neo4j"):
        try:
            return Neo4jStore()
        except Exception as exc:  # server down / bad credentials / no driver
            if prefer == "neo4j":
                raise
            print(f"[graph_store] Neo4j unavailable ({exc.__class__.__name__}); "
                  "falling back to the local mirror of movies.cypher")
    return LocalStore()
