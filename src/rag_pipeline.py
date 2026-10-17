"""End-to-end RAG pipeline: retrieve facts from the graph, then generate."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .generator import LLMGenerator, TemplateGenerator, build_prompt
from .graph_store import get_store
from .retriever import GraphRetriever


@dataclass
class RAGAnswer:
    question: str
    answer: str
    generator: str
    retrieval: object
    retrieval_ms: float
    generation_ms: float
    prompt: list


class MovieRAG:
    def __init__(self, backend="auto", provider="groq", model=None, max_facts=25):
        self.store = get_store(backend)
        self.retriever = GraphRetriever(self.store, max_facts=max_facts)
        self.llm = LLMGenerator(provider, model) if provider != "none" else None
        self.fallback = TemplateGenerator()

    def ask(self, question: str) -> RAGAnswer:
        t0 = time.perf_counter()
        r = self.retriever.retrieve(question)
        t_ret = (time.perf_counter() - t0) * 1000
        prompt = build_prompt(question, r.context())
        answer = gen = None
        t_gen = 0.0
        if self.llm is not None and self.llm.available:
            try:
                answer, gen, secs = self.llm.generate(question, r.context())
                t_gen = secs * 1000
            except Exception as exc:  # network error, bad key, rate limit ...
                gen = f"LLM error ({exc.__class__.__name__}) -> "
        if answer is None:
            answer, name, secs = self.fallback.generate(r)
            gen = (gen or "") + name
            t_gen = secs * 1000
        return RAGAnswer(question, answer, gen, r, t_ret, t_gen, prompt)
