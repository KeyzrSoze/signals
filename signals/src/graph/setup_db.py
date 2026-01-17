import os
from neo4j import GraphDatabase, basic_auth

# --- Configuration ---
# It's recommended to use environment variables for credentials in a real application.
# Example: export NEO4J_URI="bolt://localhost:7687"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Neo4jDatabase:
    """A wrapper for the Neo4j driver to manage connections and sessions."""

    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(
                uri, auth=basic_auth(user, password))
            self.driver.verify_connectivity()
            print("✅ Successfully connected to Neo4j.")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed.")

    def run_query(self, query, parameters=None):
        """Executes a write query."""
        if not self.driver:
            print("Cannot run query: Driver not initialized.")
            return
        with self.driver.session(database="neo4j") as session:
            try:
                session.run(query, parameters)
            except Exception as e:
                # This helps catch issues like constraints already existing.
                if "already exists" in str(e):
                    print(
                        f"   ⚠️  Skipping query, constraint/index likely already exists.")
                else:
                    print(f"   ❌ Query failed: {query}")
                    raise e


def create_constraints(db: Neo4jDatabase):
    """
    Creates uniqueness constraints for the graph schema to ensure data integrity
    and optimize query performance.

    Args:
        db (Neo4jDatabase): An active database connection object.
    """
    print("\nApplying database constraints...")

    # UPDATED SYNTAX FOR NEO4J 5.x
    # 1. Changed "ON" to "FOR"
    # 2. Changed "ASSERT" to "REQUIRE"
    # 3. Added "IF NOT EXISTS" to prevent errors on re-runs
    constraints = {
        "NDC": "CREATE CONSTRAINT IF NOT EXISTS FOR (n:NDC) REQUIRE n.ndc11 IS UNIQUE",
        "Ingredient": "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
        "Corporation": "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Corporation) REQUIRE c.name IS UNIQUE",
        "Facility": "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Facility) REQUIRE f.fei_number IS UNIQUE"
    }

    for name, query in constraints.items():
        print(f"   - Applying constraint for: {name}")
        db.run_query(query)

    print("✅ All constraints have been processed.")


if __name__ == "__main__":
    print("🚀 Initializing Neo4j Database Setup...")
    db_connection = None
    try:
        db_connection = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if db_connection.driver:
            create_constraints(db_connection)
    finally:
        if db_connection:
            db_connection.close()
