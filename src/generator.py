"""Answer generation: grounded prompt -> LLM, with an extractive fallback.

The LLM is reached through any OpenAI-compatible ``/chat/completions`` endpoint,
so the same code works with Groq, OpenAI, OpenRouter, or a local Ollama server.
If no provider is configured (or the call fails) a deterministic, template-based
generator verbalises the retrieved facts instead, so the app always answers.
"""
from __future__ import annotations

import os
import time

import requests

SYSTEM_PROMPT = (
    "You are a movie question-answering assistant. Answer the user's question "
    "using ONLY the facts in the CONTEXT, which were retrieved from a Neo4j "
    "movie knowledge graph. Facts have the form (subject) -[RELATION]-> (object) "
    "[detail]. Write a short, natural answer (1-3 sentences). Include release "
    "years in brackets when you mention a movie. If the context does not contain "
    "the answer, say: \"I could not find that in the movie graph.\" Never use "
    "outside knowledge."
)

PROVIDERS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1",
               "model": "llama-3.1-8b-instant", "key_env": "GROQ_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1",
               "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "model": "llama3.2", "key_env": None},
}


def build_prompt(question: str, context: str) -> list[dict]:
    user = f"CONTEXT:\n{context or '(no facts retrieved)'}\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


class LLMGenerator:
    def __init__(self, provider="groq", model=None, temperature=0.0, timeout=30):
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.base_url = os.getenv("LLM_BASE_URL", cfg["base_url"])
        self.model = model or os.getenv("LLM_MODEL", cfg["model"])
        self.api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
        self.temperature = temperature
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, question, context):
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": self.temperature,
                  "messages": build_prompt(question, context)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text, f"{self.provider}:{self.model}", time.perf_counter() - t0


def _join(items):
    items = list(dict.fromkeys(items))
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class TemplateGenerator:
    """Deterministic fallback that turns the top facts into sentences."""

    def generate(self, retrieval):
        t0 = time.perf_counter()
        return self._answer(retrieval), "template (no LLM)", time.perf_counter() - t0

    def _answer(self, r):
        facts = r.facts
        if not facts:
            return "I could not find that in the movie graph."
        by_rel = {}
        for f in facts:
            by_rel.setdefault(f.relation, []).append(f)
        wanted = list(dict.fromkeys({"ROLE": "ACTED_IN"}.get(i, i) for i in r.intents))
        q = r.question.lower()
        out = []

        if "BOTH_WORKED_ON" in by_rel:
            a_b = by_rel["BOTH_WORKED_ON"][0].subject.replace("&", "and")
            out.append(f"{a_b} both worked on "
                       f"{_join(f.obj for f in by_rel['BOTH_WORKED_ON'])}.")
        if r.strategy == "tf-idf fallback":
            top = facts[0]
            out.append(f"The closest match in the graph is {top.subject}, "
                       f"tagline: \"{top.obj.rstrip('.')}\".")
        if r.strategy == "graph (year filter)":
            y = r.years
            span = str(y[0]) if y[0] == y[1] else f"{y[0]}-{min(y[1], 2012)}"
            out.append(f"Movies released in {span}: "
                       f"{_join(f.subject + ' (' + f.obj + ')' for f in facts)}.")

        for movie in r.movies:
            mf = [f for f in facts if movie in (f.subject.split(' (')[0],
                                                f.obj.split(' (')[0])]
            label = next((f.subject for f in mf if f.relation == "RELEASED"), movie)
            for rel in wanted or ["DIRECTED", "ACTED_IN"]:
                rows = [f for f in mf if f.relation == rel]
                if not rows:
                    continue
                if rel == "RELEASED":
                    out.append(f"{movie} was released in {rows[0].obj}.")
                elif rel == "TAGLINE":
                    out.append(f"The tagline of {label} is \"{rows[0].obj.rstrip('.')}\".")
                elif rel == "ACTED_IN":
                    if r.people:
                        rows = [f for f in rows if f.subject in r.people] or rows
                    if "ROLE" in r.intents:   # "who played Neo" -> only Neo
                        rows = [f for f in rows if any(
                            role.lower() in q for role in
                            f.detail[3:].split(", ") if role)] or rows
                    cast = [f"{f.subject} ({f.detail})" if f.detail else f.subject
                            for f in rows]
                    out.append(f"In {label}, the cast includes {_join(cast)}.")
                elif rel == "REVIEWED":
                    out.append(f"{label} was reviewed by " + _join(
                        f"{f.subject} ({f.detail})" for f in rows) + ".")
                else:
                    verb = {"DIRECTED": "directed", "WROTE": "written",
                            "PRODUCED": "produced"}[rel]
                    out.append(f"{label} was {verb} by "
                               f"{_join(f.subject for f in rows)}.")

        if not r.movies and "BOTH_WORKED_ON" not in by_rel:
            for person in r.people:
                pf = [f for f in facts if f.subject == person]
                for rel in wanted or ["ACTED_IN", "DIRECTED"]:
                    rows = sorted((f for f in pf if f.relation == rel),
                                  key=lambda f: f.obj[-5:-1])
                    if not rows:
                        continue
                    if rel == "BORN":
                        out.append(f"{person} was born in {rows[0].obj}.")
                        continue
                    verb = {"ACTED_IN": "acted in", "DIRECTED": "directed",
                            "WROTE": "wrote", "PRODUCED": "produced",
                            "REVIEWED": "reviewed"}.get(rel)
                    if verb:
                        items = [f"{f.obj} {f.detail}".strip() if rel == "ACTED_IN"
                                 and "ROLE" in r.intents else f.obj for f in rows]
                        out.append(f"{person} {verb} {_join(items)}.")

        if not out:   # entity found but nothing matched the intent
            out.append("From the graph: " + "; ".join(
                f"{f.subject} {f.relation.lower().replace('_', ' ')} {f.obj}"
                for f in facts[:5]) + ".")
        return " ".join(out)
