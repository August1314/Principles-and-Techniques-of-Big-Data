from __future__ import annotations

import argparse
import csv
from pathlib import Path

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
NODES_CSV = ROOT / "outputs" / "nodes.csv"
EDGES_CSV = ROOT / "outputs" / "edges.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", required=True)
    p.add_argument("--database", default="neo4j")
    args = p.parse_args()

    if not NODES_CSV.exists() or not EDGES_CSV.exists():
        raise FileNotFoundError("请先运行 extract_kg.py 生成 outputs/nodes.csv 与 outputs/edges.csv")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    with driver.session(database=args.database) as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

        # nodes
        with NODES_CSV.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n.name = $name
                    SET n:`%s`
                    """
                    % row["label"],
                    id=row["id"],
                    name=row["name"],
                )

        # edges
        with EDGES_CSV.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                session.run(
                    """
                    MATCH (a:Entity {id: $source})
                    MATCH (b:Entity {id: $target})
                    MERGE (a)-[r:%s]->(b)
                    SET r.evidence = $evidence
                    """
                    % row["rel"],
                    source=row["source"],
                    target=row["target"],
                    evidence=row.get("evidence", ""),
                )

    driver.close()
    print("Loaded into Neo4j OK.")


if __name__ == "__main__":
    main()

