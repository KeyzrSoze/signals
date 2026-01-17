import os
import polars as pl
from neo4j import GraphDatabase, basic_auth
import rapidfuzz
import google.generativeai as genai
import json
import logging
import re
from typing import List, Dict

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Data paths and URLs
FDA_URL = "https://www.accessdata.fda.gov/downloads/drug/drd/drug-establishments-current-registration-site.zip"
FDA_DATA_PATH = "signals/data/raw/fda_establishments.csv"

# Fuzzy matching thresholds
FUZZ_ACCEPT_THRESHOLD = 90
FUZZ_REJECT_THRESHOLD = 50
LLM_BATCH_SIZE = 10

# --- Neo4j Operations ---

def get_subsidiaries_from_graph(driver) -> pl.DataFrame:
    """Fetches all Subsidiary names from the Neo4j database."""
    logging.info("Fetching Subsidiary names from Neo4j...")
    query = "MATCH (s:Subsidiary) WHERE s.id IS NOT NULL RETURN s.id AS subsidiary_name"
    with driver.session(database="neo4j") as session:
        result = session.run(query)
        data = [record["subsidiary_name"] for record in result]
        logging.info(f"Found {len(data)} subsidiaries.")
        return pl.DataFrame({"subsidiary_name": data})

def link_subsidiary_to_facility(driver, links: List[Dict]):
    """Creates an [:OPERATES] relationship between a Subsidiary and a Facility."""
    if not links:
        logging.warning("No new links to create.")
        return
    logging.info(f"Writing {len(links)} new facility links to the graph...")
    query = """
    UNWIND $links AS link
    MERGE (s:Subsidiary {id: link.subsidiary_name})
    MERGE (f:Facility {fei_number: link.FEI_NUMBER})
        ON CREATE SET f.name = link.FIRM_NAME, f.address = link.FIRM_ADDRESS
    MERGE (s)-[:OPERATES]->(f)
    """
    with driver.session(database="neo4j") as session:
        session.run(query, links=links)
    logging.info("Successfully wrote links to Neo4j.")

# --- Step 1: Ingestion ---

def prepare_fda_data(url: str, local_path: str) -> pl.DataFrame:
    """
    Ensures FDA data is available locally (stubbed download) and loads it.
    """
    if not os.path.exists(local_path):
        logging.warning(f"FDA data not found at {local_path}. This is a stub.")
        logging.info(f"In a real run, download and unzip from: {url}")
        # Create a dummy file for demonstration purposes
        dummy_df = pl.DataFrame({
            "FEI_NUMBER": ["1234567", "2345678", "3456789", "4567890"],
            "Firm_Name": ["PFIZER PHARMACEUTICALS LLC", "SANDOZ INC", "RANDOM LABS", "TEVA"],
            "Street": ["123 Main St", "456 Oak Ave", "789 Pine Ln", "101 Maple Rd"],
            "City": ["Anytown", "Someville", "Otherplace", "New City"],
            "State": ["CA", "NY", "TX", "FL"],
            "Zip_Code": ["12345", "67890", "54321", "09876"],
            "Country_Code": ["USA", "USA", "USA", "USA"]
        })
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        dummy_df.write_csv(local_path)
        logging.info(f"Created a dummy FDA data file at: {local_path}")
    
    logging.info(f"Loading FDA data from {local_path}...")
    df = pl.read_csv(local_path, dtypes={"FEI_NUMBER": pl.Utf8})
    df = df.with_columns(
        pl.concat_str([
            pl.col("Street"), pl.col("City"), pl.col("State"), pl.col("Zip_Code"), pl.col("Country_Code")
        ], separator=", ").alias("FIRM_ADDRESS")
    )
    return df.select(["FEI_NUMBER", "Firm_Name", "FIRM_ADDRESS"])

# --- Step 2: Blocking Filter ---

def normalize_name(name: str) -> str:
    """A simple function to normalize company names for better matching."""
    if not name: return ""
    name = name.upper()
    name = re.sub(r"[,.\s](LLC|INC|CORP|LTD|LP)$", "", name)
    name = re.sub(r"[^A-Z0-9\s]", "", name)
    return name.strip()

def fuzzy_match_entities(subs_df: pl.DataFrame, fda_df: pl.DataFrame) -> (pl.DataFrame, pl.DataFrame):
    """
    Performs fuzzy matching and applies a blocking strategy to categorize matches.
    """
    logging.info("Performing fuzzy matching between subsidiaries and FDA firms...")
    
    # Create normalized join keys
    subs_df = subs_df.with_columns(pl.col("subsidiary_name").apply(normalize_name).alias("key"))
    fda_df = fda_df.with_columns(pl.col("Firm_Name").apply(normalize_name).alias("key"))

    # Join on the normalized key to create candidate pairs
    candidates = subs_df.join(fda_df, on="key")
    
    # Calculate similarity score for the original names
    scores = rapidfuzz.process.cdist(
        candidates["subsidiary_name"].to_list(),
        candidates["Firm_Name"].to_list(),
        scorer=rapidfuzz.fuzz.WRatio,
        workers=-1
    )
    
    # Since we joined on key, we get a diagonal matrix of scores for the pairs
    candidates = candidates.with_columns(pl.Series("score", np.diag(scores)))

    logging.info(f"Scored {len(candidates)} potential matches.")

    # Apply blocking strategy
    auto_accept = candidates.filter(pl.col("score") > FUZZ_ACCEPT_THRESHOLD)
    gray_zone = candidates.filter(
        (pl.col("score") >= FUZZ_REJECT_THRESHOLD) & (pl.col("score") <= FUZZ_ACCEPT_THRESHOLD)
    )
    
    logging.info(f"Found {len(auto_accept)} auto-accept matches.")
    logging.info(f"Found {len(gray_zone)} gray-zone matches for LLM resolution.")
    
    return auto_accept, gray_zone

# --- Step 3: LLM Resolution ---

def get_llm_verdicts(gray_zone_df: pl.DataFrame) -> List[Dict]:
    """
    Sends batches of gray-zone matches to Gemini for verification.
    """
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set. Cannot resolve gray zone matches.")
        return []
        
    if gray_zone_df.is_empty():
        return []

    logging.info(f"Resolving {len(gray_zone_df)} gray-zone matches with Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    confirmed_links = []

    for i in range(0, len(gray_zone_df), LLM_BATCH_SIZE):
        batch = gray_zone_df[i:i + LLM_BATCH_SIZE]
        
        prompt_pairs = [
            {"database_name": row["subsidiary_name"], "fda_registry_name": row["Firm_Name"]}
            for row in batch.to_dicts()
        ]

        prompt = f"""
        You are an expert in pharmaceutical supply chain analysis. Your task is to determine if two company names refer to the same entity.

        Instructions:
        1. Analyze the pairs of names provided below. One name is from our internal database, and the other is from the official FDA registry.
        2. Consider common variations like 'Inc', 'LLC', 'Corp', abbreviations, and potential typos.
        3. For each pair, return 'true' if they are the same company, and 'false' otherwise.
        4. Your final output MUST be a single, valid JSON list of booleans, corresponding to the order of the pairs. Do not include any other text or explanation.

        Here are the pairs:
        {json.dumps(prompt_pairs, indent=2)}
        """
        
        try:
            response = model.generate_content(prompt)
            # Clean and parse the JSON response
            cleaned_response = re.search(r"\[.*\]", response.text, re.DOTALL).group(0)
            verdicts = json.loads(cleaned_response)

            for idx, verdict in enumerate(verdicts):
                if verdict is True:
                    confirmed_links.append(batch[idx].to_dicts()[0])
        except Exception as e:
            logging.error(f"An error occurred during LLM resolution for batch {i//LLM_BATCH_SIZE + 1}: {e}")

    logging.info(f"LLM confirmed {len(confirmed_links)} additional matches.")
    return confirmed_links

# --- Main Orchestration ---

def main():
    """Main function to run the facility enrichment process."""
    logging.info("🚀 Starting facility enrichment process...")
    driver = None
    try:
        # Connect to services
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        
        # Step 1: Get data
        subsidiaries_df = get_subsidiaries_from_graph(driver)
        fda_df = prepare_fda_data(FDA_URL, FDA_DATA_PATH)

        # Step 2: Fuzzy match and block
        auto_accept_df, gray_zone_df = fuzzy_match_entities(subsidiaries_df, fda_df)
        
        # Write auto-accepted matches to Neo4j
        if not auto_accept_df.is_empty():
            link_subsidiary_to_facility(driver, auto_accept_df.to_dicts())

        # Step 3: LLM Resolution
        llm_confirmed_links = get_llm_verdicts(gray_zone_df)
        if llm_confirmed_links:
            link_subsidiary_to_facility(driver, llm_confirmed_links)

        logging.info("✅ Facility enrichment process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the enrichment process: {e}")
    finally:
        if driver:
            driver.close()
            logging.info("🔌 Neo4j connection closed.")

if __name__ == "__main__":
    # Ensure you have a GEMINI_API_KEY environment variable set to run the LLM part
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY environment variable not found. LLM resolution will be skipped.")
    main()
