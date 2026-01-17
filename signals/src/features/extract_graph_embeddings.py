import os
import polars as pl
from neo4j import GraphDatabase, basic_auth
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# GDS and Feature Config
GDS_PROJECTION_NAME = "supply_chain_graph"
EMBEDDING_VECTOR_SIZE = 16
OUTPUT_PATH = "signals/data/processed/graph_features.parquet"


class GraphFeatureExtractor:
    """
    Extracts topological features (embeddings, centrality) from the Neo4j graph.
    """

    def __init__(self, uri: str, user: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self.driver.verify_connectivity()
            logging.info("✅ Successfully connected to Neo4j.")
        except Exception as e:
            logging.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            logging.info("🔌 Neo4j connection closed.")

    def _run_query(self, query: str, params: dict = None) -> list:
        """Helper to run a query and return a list of records."""
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def _ensure_gds_projection(self) -> bool:
        """
        Checks if the GDS graph projection exists and creates it if not.
        NOTE: Requires the Neo4j Graph Data Science plugin to be installed.
        """
        logging.info(f"Checking for GDS projection: '{GDS_PROJECTION_NAME}'...")
        
        # Check if GDS is installed by trying a basic command
        try:
            self._run_query("RETURN gds.version()")
        except Exception as e:
            if "Unknown function 'gds.version'" in str(e):
                logging.error("❌ GDS plugin not found on the Neo4j server.")
                logging.error("   Cannot generate graph embeddings. Please install the GDS plugin.")
                return False
            # Some other error
            raise e
            
        exists_query = "CALL gds.graph.exists($name) YIELD exists"
        if not self._run_query(exists_query, params={"name": GDS_PROJECTION_NAME})[0]['exists']:
            logging.info("Projection not found. Creating a new one...")
            # Create a projection of the full graph topology
            create_query = """
            CALL gds.graph.project(
                $name, 
                ['NDC', 'Ingredient', 'Subsidiary', 'Corporation', 'Facility'],
                {
                    OWNS: {orientation: 'UNDIRECTED'},
                    MARKETS: {orientation: 'UNDIRECTED'},
                    CONTAINS: {orientation: 'UNDIRECTED'},
                    OPERATES: {orientation: 'UNDIRECTED'}
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            """
            result = self._run_query(create_query, params={"name": GDS_PROJECTION_NAME})[0]
            logging.info(f"   Created projection with {result['nodeCount']} nodes and {result['relationshipCount']} relationships.")
        else:
            logging.info("   Projection already exists.")
        return True

    def _get_fastrp_embeddings(self) -> pl.DataFrame:
        """
        Generates graph embeddings for NDC nodes using the FastRP algorithm.
        """
        logging.info("Generating FastRP embeddings for NDC nodes...")
        query = """
        CALL gds.fastRp.stream($name, {
            embeddingDimension: $dim,
            nodeLabels: ['NDC']
        })
        YIELD nodeId, embedding
        WITH gds.util.asNode(nodeId) AS n, embedding
        RETURN n.ndc11 AS ndc11, embedding AS graph_embedding_vector
        """
        try:
            results = self._run_query(query, params={"name": GDS_PROJECTION_NAME, "dim": EMBEDDING_VECTOR_SIZE})
            if not results:
                logging.warning("FastRP did not return any embeddings.")
                return pl.DataFrame({"ndc11": [], "graph_embedding_vector": []})
            return pl.from_records(results)
        except Exception as e:
            logging.error(f"❌ Failed to run FastRP: {e}")
            logging.error("   This may be because the GDS plugin is not installed or configured correctly.")
            return None

    def _get_supplier_diversity(self) -> pl.DataFrame:
        """
        Calculates a diversity score based on the number of facilities producing an NDC's ingredients.
        """
        logging.info("Calculating supplier diversity scores...")
        query = """
        MATCH (i:Ingredient)<-[:CONTAINS]-(n:NDC)
        // OPTIONAL MATCH handles ingredients that might not have a facility link yet
        OPTIONAL MATCH (i)<-[:CONTAINS]-(:NDC)<-[:MARKETS]-(:Subsidiary)-[:OPERATES]->(f:Facility)
        WITH n, count(DISTINCT f) AS facility_count
        RETURN n.ndc11 AS ndc11, facility_count AS supplier_diversity_score
        """
        results = self._run_query(query)
        return pl.from_records(results)

    def extract_features(self) -> pl.DataFrame:
        """
        Orchestrates the feature extraction process.
        """
        if not self.driver:
            return None
        
        # 1. Generate embeddings with GDS
        if not self._ensure_gds_projection():
            return None # Stop if GDS is not available
        embeddings_df = self._get_fastrp_embeddings()
        if embeddings_df is None:
            return None # Stop if embedding generation failed

        # 2. Calculate supplier diversity
        diversity_df = self._get_supplier_diversity()

        # 3. Join features and return
        logging.info("Joining embedding and diversity features...")
        final_df = diversity_df.join(embeddings_df, on="ndc11", how="left")
        
        return final_df


def main():
    """Main function to run the graph feature extraction."""
    logging.info("🚀 Starting graph feature extraction process...")
    extractor = None
    try:
        extractor = GraphFeatureExtractor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        graph_features = extractor.extract_features()

        if graph_features is not None and not graph_features.is_empty():
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            graph_features.write_parquet(OUTPUT_PATH)
            logging.info(f"✅ Successfully saved {len(graph_features)} records to '{OUTPUT_PATH}'")
        else:
            logging.warning("No graph features were generated. Output file was not created.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if extractor:
            extractor.close()

if __name__ == "__main__":
    main()
