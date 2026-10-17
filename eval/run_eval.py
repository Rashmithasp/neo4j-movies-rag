"""Evaluate the Movie RAG pipeline on eval/questions.json.

Metrics per question
  linking_ok   : linked entities == expected entities (order-free)
  context_hit  : every gold string appears in the retrieved context
  answer_hit   : every gold string appears in the generated answer
                 (for out-of-graph questions: the answer declines)

Usage:  python -m eval.run_eval [--backend local|neo4j|auto] [--provider none|groq|openai|ollama]
"""
import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from src.rag_pipeline import MovieRAG

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--provider", default="groq")
    args = ap.parse_args()

    rag = MovieRAG(backend=args.backend, provider=args.provider)
    items = json.loads((HERE / "questions.json").read_text())
    rows = []
    for it in items:
        a = rag.ask(it["q"])
        r = a.retrieval
        ctx, ans = r.context().lower(), a.answer.lower()
        linked = sorted(r.movies + r.people)
        if it["gold"] == ["__NOT_FOUND__"]:
            ctx_hit = not r.facts
            ans_hit = "could not find" in ans
        else:
            ctx_hit = all(g.lower() in ctx for g in it["gold"])
            ans_hit = all(g.lower() in ans for g in it["gold"])
        rows.append({
            "id": it["id"], "type": it["type"], "question": it["q"],
            "linked": "; ".join(linked) or "-",
            "intents": ", ".join(r.intents) or "-",
            "strategy": r.strategy, "facts": len(r.facts),
            "linking_ok": linked == sorted(it["entities"]),
            "context_hit": ctx_hit, "answer_hit": ans_hit,
            "retrieval_ms": round(a.retrieval_ms, 2),
            "prompt_chars": sum(len(m["content"]) for m in a.prompt),
            "generator": a.generator, "answer": a.answer,
        })
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False)

    summary = {
        "backend": rag.store.name, "generator": rows[0]["generator"],
        "questions": len(df),
        "entity_linking_acc": round(df.linking_ok.mean(), 3),
        "context_recall": round(df.context_hit.mean(), 3),
        "answer_accuracy": round(df.answer_hit.mean(), 3),
        "mean_facts": round(df.facts.mean(), 1),
        "mean_retrieval_ms": round(df.retrieval_ms.mean(), 2),
        "median_retrieval_ms": round(statistics.median(df.retrieval_ms), 2),
        "mean_prompt_chars": round(df.prompt_chars.mean()),
        "by_type": df.groupby("type", sort=False)[["context_hit", "answer_hit"]]
                     .mean().round(2).to_dict("index"),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(df[["id", "type", "linked", "strategy", "facts", "linking_ok",
              "context_hit", "answer_hit"]].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_type"}, indent=2))


if __name__ == "__main__":
    main()
