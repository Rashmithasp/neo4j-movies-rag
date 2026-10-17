"""Load data/movies.cypher into a running Neo4j database.

Usage:  python scripts/load_movies.py   (reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)
"""
import os
from pathlib import Path

from neo4j import GraphDatabase

SCRIPT = Path(__file__).resolve().parent.parent / "data" / "movies.cypher"

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
)
with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
    for stmt in SCRIPT.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            s.run(stmt)
    n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    r = s.run("MATCH ()-[r]->() RETURN count(r) AS r").single()["r"]
print(f"Loaded Movies graph: {n} nodes, {r} relationships")
driver.close()
