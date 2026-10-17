# Neo4j Movies Graph RAG

Retrieval-Augmented Generation over the Neo4j Movies knowledge graph: questions are linked to
movies/people in the graph, the relevant subgraph is retrieved with Cypher, and an LLM answers
using only the retrieved facts. Advance NLP (AML23702), Lab Exercise 2.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in Neo4j password and an LLM key, then export the variables
```

1. Start Neo4j (Neo4j Desktop, Docker `neo4j:5`, or a free Aura instance).
2. Load the dataset: `python scripts/load_movies.py` → `Loaded Movies graph: 171 nodes, 253 relationships`
3. Get a free Groq API key (https://console.groq.com) and set `GROQ_API_KEY`
   (or use `OPENAI_API_KEY`, or run Ollama locally and pick "Ollama" in the sidebar).
4. Run the app: `streamlit run app.py`

If Neo4j is not reachable, "Auto" mode falls back to an in-memory mirror built from the same
`data/movies.cypher`. If no LLM key is set, answers come from a template generator.

## Evaluate

```bash
python -m eval.run_eval --backend neo4j --provider groq
```

Writes `eval/results.csv` (per question) and `eval/summary.json` (aggregates).

## Layout

```
app.py                  Streamlit UI
src/graph_store.py      Neo4jStore (Cypher) + LocalStore (offline mirror)
src/retriever.py        entity linking, intent detection, graph retrieval, TF-IDF fallback
src/generator.py        grounded prompt, OpenAI-compatible LLM client, template fallback
src/rag_pipeline.py     MovieRAG: retrieve -> generate
scripts/load_movies.py  loads data/movies.cypher into Neo4j
eval/                   30-question test set and evaluation script
```
