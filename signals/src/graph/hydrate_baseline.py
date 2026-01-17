import polars as pl
from neo4j import GraphDatabase, basic_auth
import os
import time

# --- Configuration ---
# Match these with your docker-compose environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# Use the password you set in docker-compose
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DATA_PATH = "data/processed/ndc_entity_map.parquet"

# --- Cypher Query ---
# MERGE ensures we don't create duplicates.
# We create the entire chain: Corporation -> Subsidiary -> NDC -> Ingredient
INGEST_QUERY = """
UNWIND $batch AS row
MERGE (corp:Corporation {name: row.manufacturer})
MERGE (sub:Subsidiary {labeler_id: row.labeler_id})
MERGE (corp)-[:OWNS]->(sub)

MERGE (ing:Ingredient {name: row.ingredient})

MERGE (n:NDC {ndc11: row.ndc11})
SET n.description = row.drug_description
MERGE (sub)-[:MARKETS]->(n)
MERGE (n)-[:CONTAINS]->(ing)
"""


def ingest_batch(tx, batch_data):
    """
    Transaction function to execute the batch ingest.
    """
    tx.run(INGEST_QUERY, batch=batch_data)


def hydrate_graph():
    print("🚀 Starting graph hydration process...")

    # 1. Load Data
    try:
        df = pl.read_parquet(DATA_PATH)
        print(f"✅ Loaded {df.height:,} records from '{DATA_PATH}'.")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. Connect to Neo4j
    driver = None
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected to Neo4j database.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 3. Ingest in Batches
    BATCH_SIZE = 1000
    total_rows = df.height

    # Convert Polars DataFrame to a list of dictionaries for Neo4j
    # (Polars .to_dicts() is fast and memory efficient)
    records = df.to_dicts()

    print(f"   Ingesting data in batches of {BATCH_SIZE}...")
    start_time = time.time()

    with driver.session(database="neo4j") as session:
        for i in range(0, total_rows, BATCH_SIZE):
            batch = records[i: i + BATCH_SIZE]

            try:
                # UPDATED FOR NEO4J 5.x: write_transaction -> execute_write
                session.execute_write(ingest_batch, batch)

                # Progress bar effect
                if (i // BATCH_SIZE) % 10 == 0:
                    print(
                        f"      Processed {i + len(batch):,} / {total_rows:,} rows...", end="\r")
            except Exception as e:
                print(f"\n❌ Error on batch starting at index {i}: {e}")
                break

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"\n✅ Hydration Complete! Processed {total_rows:,} nodes in {duration:.2f} seconds.")

    driver.close()
    print("🔌 Connection to Neo4j closed.")


if __name__ == "__main__":
    hydrate_graph()
