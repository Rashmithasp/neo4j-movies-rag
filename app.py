"""Streamlit front end for the Neo4j Movies RAG system.

Run:  streamlit run app.py
"""
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.graph_store import CYPHER
from src.rag_pipeline import MovieRAG

st.set_page_config(page_title="Movie Graph RAG", page_icon="🎬", layout="wide")

EXAMPLES = [
    "Who directed The Matrix?",
    "Who played Neo in The Matrix?",
    "What movies have Tom Hanks and Meg Ryan done together?",
    "Which movies did Tom Hanks act in after 2000?",
    "What rating did Jessica Thompson give Unforgiven?",
    "Which movies were released in 1999?",
    "Which movie has the tagline Freedom! Forever!?",
]
REL_COLORS = {"ACTED_IN": "#4C78A8", "DIRECTED": "#E45756", "WROTE": "#72B7B2",
              "PRODUCED": "#F58518", "REVIEWED": "#B279A2"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")
    backend = st.selectbox("Knowledge-graph back end",
                           ["auto", "neo4j", "local"],
                           format_func={"auto": "Auto (Neo4j, else local)",
                                        "neo4j": "Neo4j server",
                                        "local": "Local mirror (offline)"}.get)
    provider = st.selectbox("LLM provider", ["groq", "openai", "ollama", "none"],
                            format_func={"groq": "Groq (Llama 3.1 8B)",
                                         "openai": "OpenAI (gpt-4o-mini)",
                                         "ollama": "Ollama (local)",
                                         "none": "None (template answers)"}.get)
    max_facts = st.slider("Max facts in context", 5, 50, 25, 5)


@st.cache_resource(show_spinner="Connecting to the movie graph…")
def load_rag(backend, provider, max_facts):
    return MovieRAG(backend=backend, provider=provider, max_facts=max_facts)


rag = load_rag(backend, provider, max_facts)
stats = rag.store.stats()
with st.sidebar:
    st.divider()
    st.caption(f"Connected to: **{rag.store.name}**")
    c1, c2 = st.columns(2)
    c1.metric("Nodes", stats["nodes"])
    c2.metric("Relationships", stats["rels"])
    st.caption(f"{len(rag.retriever.titles)} movies · {len(rag.retriever.people)} people")
    llm_ok = rag.llm is not None and rag.llm.available
    st.caption("LLM: " + (f"**{rag.llm.provider}:{rag.llm.model}**" if llm_ok
                          else "not configured → template answers"))

# ---------------------------------------------------------------- main
st.title("🎬 Movie Knowledge-Graph RAG")
st.markdown("Ask a question about the **Neo4j Movies** graph. Relevant facts are "
            "retrieved from the graph and passed to an LLM, which writes the answer.")
st.divider()

example = st.selectbox("Try an example question", ["—"] + EXAMPLES)
question = st.text_input("Your question",
                         value="" if example == "—" else example,
                         placeholder="e.g. Who directed The Matrix?")
ask = st.button("🔎 Ask", type="primary")

if ask and question.strip():
    res = rag.ask(question.strip())
    r = res.retrieval
    st.subheader("Answer")
    st.success(res.answer)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Facts retrieved", len(r.facts))
    m2.metric("Retrieval", f"{res.retrieval_ms:.1f} ms")
    m3.metric("Generation", f"{res.generation_ms:.0f} ms")
    m4.metric("Strategy", r.strategy.split(" (")[0])
    st.caption(f"Generator: {res.generator} · Linked movies: "
               f"{', '.join(r.movies) or '—'} · Linked people: "
               f"{', '.join(r.people) or '—'} · Intents: "
               f"{', '.join(r.intents) or '—'}"
               + (f" · Years: {r.years[0]}–{r.years[1]}" if r.years else ""))

    t1, t2, t3, t4 = st.tabs(["Retrieved facts", "Subgraph", "LLM prompt", "Cypher"])
    df = pd.DataFrame([{"Subject": f.subject, "Relation": f.relation,
                        "Object": f.obj, "Detail": f.detail,
                        "Score": round(f.score, 2)} for f in r.facts])
    with t1:
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not df.empty:
            st.download_button("Download facts (CSV)", df.to_csv(index=False),
                               "retrieved_facts.csv", "text/csv")
    with t2:
        rels = df[df.Relation.isin(REL_COLORS)] if not df.empty else df
        if rels.empty:
            st.info("No person–movie edges to draw for this question.")
        else:
            G = nx.DiGraph()
            for _, row in rels.iterrows():
                G.add_edge(row.Subject, row.Object, rel=row.Relation)
            pos = nx.spring_layout(G, seed=7, k=0.9)
            fig = go.Figure()
            for rel, color in REL_COLORS.items():
                xs, ys = [], []
                for a, b, d in G.edges(data=True):
                    if d["rel"] == rel:
                        xs += [pos[a][0], pos[b][0], None]
                        ys += [pos[a][1], pos[b][1], None]
                if xs:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=rel,
                                             line=dict(color=color, width=2)))
            shared = set(df[df.Relation == "BOTH_WORKED_ON"].Object)
            movies = {n for n in G if n.endswith(")")}
            for group, sym, col in [(movies - shared, "square", "#888"),
                                    (movies & shared, "square", "#54A24B"),
                                    (set(G) - movies, "circle", "#FFB000")]:
                fig.add_trace(go.Scatter(
                    x=[pos[n][0] for n in group], y=[pos[n][1] for n in group],
                    mode="markers+text", text=list(group), textposition="top center",
                    marker=dict(size=14 if col != "#54A24B" else 20, symbol=sym,
                                color=col), showlegend=False))
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)
    with t3:
        for msg in res.prompt:
            st.markdown(f"**{msg['role']}**")
            st.code(msg["content"], language="text")
    with t4:
        for key in dict.fromkeys(r.queries):
            st.markdown(f"`{key}`")
            st.code(CYPHER[key].strip(), language="cypher")
elif ask:
    st.warning("Please type a question first.")

st.divider()
st.caption("Neo4j Movies RAG · Python · Neo4j · Streamlit · scikit-learn · "
           "RapidFuzz · NetworkX · Plotly")
