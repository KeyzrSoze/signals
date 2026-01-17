import os
from neo4j import GraphDatabase, basic_auth
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Neo4j connection details from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class RiskEngine:
    """
    A class to manage and propagate risk shockwaves through the Neo4j graph.
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initializes the RiskEngine and connects to the Neo4j database.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self.driver.verify_connectivity()
            logging.info("✅ Successfully connected to Neo4j.")
        except Exception as e:
            logging.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        """Closes the connection to the database."""
        if self.driver:
            self.driver.close()
            logging.info("🔌 Neo4j connection closed.")

    def propagate_factory_failure(self, fei_number: str, severity_score: float):
        """
        Propagates a risk score from a failed facility to connected NDCs.

        Args:
            fei_number (str): The FEI number of the facility that is the source of the risk.
            severity_score (float): The initial risk score (0-10).
        """
        if not self.driver:
            logging.error("Cannot propagate risk: Driver not initialized.")
            return

        logging.info(f"💥 Propagating shockwave from Facility FEI: {fei_number} with severity: {severity_score}")
        
        with self.driver.session(database="neo4j") as session:
            # Step 1: Direct propagation (1-hop business logic)
            direct_summary = session.write_transaction(
                self._direct_propagation_tx, fei_number, severity_score
            )
            logging.info(f"   - Directly affected {direct_summary.counters.properties_set} NDC(s).")

            # Step 2: Indirect propagation (2-hop business logic)
            indirect_summary = session.write_transaction(
                self._indirect_propagation_tx, fei_number, severity_score
            )
            logging.info(f"   - Indirectly affected {indirect_summary.counters.properties_set} NDC(s) via financial contagion.")
        
        logging.info("✅ Shockwave propagation complete.")

    @staticmethod
    def _direct_propagation_tx(tx, fei_number: str, severity: float):
        """
        Transaction to apply risk to NDCs directly manufactured at the facility.
        (Distance = 1 hop: Facility -> Subsidiary -> NDC)
        """
        query = """
        MATCH (f:Facility {fei_number: $fei_number})<-[:OPERATES]-(s:Subsidiary)-[:MARKETS]->(n:NDC)
        SET n.latest_risk_score = $severity,
            n.risk_source = 'Direct Facility Failure: ' + $fei_number
        RETURN count(n)
        """
        result = tx.run(query, fei_number=fei_number, severity=severity)
        return result.consume()

    @staticmethod
    def _indirect_propagation_tx(tx, fei_number: str, severity: float):
        """
        Transaction to apply decayed risk to NDCs from sibling subsidiaries.
        (Distance = 2 hops: Facility -> Sub -> Corp -> Other Sub -> NDC)
        """
        decay_factor = 0.3
        indirect_severity = severity * decay_factor

        query = """
        MATCH (f:Facility {fei_number: $fei_number})<-[:OPERATES]-(s1:Subsidiary)<-[:OWNS]-(:Corporation)-[:OWNS]->(s2:Subsidiary)
        WHERE s1 <> s2
        MATCH (s2)-[:MARKETS]->(n:NDC)
        
        // Apply risk only if it's a new risk or a higher risk than the existing one
        WHERE n.latest_risk_score IS NULL OR n.latest_risk_score < $indirect_severity
        
        SET n.latest_risk_score = $indirect_severity,
            n.risk_source = 'Financial Contagion from Facility: ' + $fei_number
        RETURN count(n)
        """
        result = tx.run(query, fei_number=fei_number, indirect_severity=indirect_severity)
        return result.consume()


if __name__ == '__main__':
    logging.info("🚀 Running RiskEngine example...")
    
    # This is an example of how to use the RiskEngine.
    # It assumes a Neo4j instance is running and has been hydrated with data.
    risk_engine = RiskEngine(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    if risk_engine.driver:
        try:
            # Simulate a failure at a facility with FEI Number '1234567'
            # This FEI is from the dummy data in enrich_facilities.py
            test_fei_number = "1234567"
            test_severity = 9.0
            risk_engine.propagate_factory_failure(test_fei_number, test_severity)

        except Exception as e:
            logging.error(f"An error occurred in the example run: {e}")
        finally:
            risk_engine.close()
    else:
        logging.error("Could not run example because Neo4j connection failed.")
