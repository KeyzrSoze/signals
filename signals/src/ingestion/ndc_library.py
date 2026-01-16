import requests
import polars as pl
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
# We use the 'drug/ndc' endpoint to get the master list of drugs
API_URL = "https://api.fda.gov/drug/ndc.json"
PROCESSED_DATA_PATH = "data/processed"


def fetch_ndc_directory():
    """
    Fetches the master NDC directory from OpenFDA.
    This gives us the Link between NDC <-> Ingredient <-> Manufacturer
    """
    print("🚀 Starting NDC Directory Ingestion...")

    all_products = []
    skip = 0
    limit = 1000

    # We'll fetch a safe chunk of the directory (e.g., 30k records)
    # For a full enterprise run, we would loop until exhaustion,
    # but for this setup, we'll grab enough to build a solid map.
    # (The full DB is large, so we loop until empty or hit a safety cap)
    max_records = 50000

    while len(all_products) < max_records:
        params = {
            "limit": limit,
            "skip": skip
        }

        try:
            # print(f"   📡 Fetching NDCs {skip} to {skip + limit}...")
            response = requests.get(API_URL, params=params)
            data = response.json()

            if "error" in data:
                break

            results = data.get("results", [])
            if not results:
                break

            all_products.extend(results)
            skip += limit

            # Rate limiting
            time.sleep(0.5)

            # Progress update every 5k
            if len(all_products) % 5000 == 0:
                print(f"      ...fetched {len(all_products)} records so far")

        except Exception as e:
            print(f"   ❌ Network Error: {e}")
            break

    print(f"   📥 Total NDCs Fetched: {len(all_products)}")
    return all_products


def process_ndc_directory(raw_data):
    """
    Clean and Normalize the NDC Data.
    Crucial Step: Convert 10-digit NDCs to 11-digit format if possible,
    or at least store the raw segments for matching.
    """
    print("   ⚙️  Building Entity Map...")

    processed_rows = []

    for item in raw_data:
        # Extract Core Identifiers
        product_ndc = item.get("product_ndc", "")  # e.g. "0591-2897"
        generic_name = item.get("generic_name", "UNKNOWN").upper()
        brand_name = item.get("brand_name", "UNKNOWN").upper()
        labeler_name = item.get("labeler_name", "UNKNOWN").upper()

        # Extract Active Ingredients (Usually a list)
        ingredients = item.get("active_ingredients", [])
        ingredient_name = ingredients[0].get(
            "name", "UNKNOWN").upper() if ingredients else generic_name

        processed_rows.append({
            "product_ndc": product_ndc,
            "generic_name": generic_name,
            "brand_name": brand_name,
            "labeler_name": labeler_name,
            "ingredient_name": ingredient_name,
            "marketing_start_date": item.get("marketing_start_date"),
            "marketing_end_date": item.get("marketing_end_date"),
            "product_type": item.get("product_type")
        })

    df = pl.DataFrame(processed_rows)

    if df.is_empty():
        return None

    # Normalization Logic
    df = (
        df
        .with_columns([
            # Create a "clean" manufacturer name (simplify LLC, Inc, etc)
            pl.col("labeler_name")
              .str.replace(" INC", "")
              .str.replace(" LLC", "")
              .str.replace(" CORP", "")
              .str.replace(" LTD", "")
              .str.strip_chars()
              .alias("manufacturer_simple"),

            # Parse Dates
            pl.col("marketing_start_date").str.strptime(
                pl.Date, "%Y%m%d", strict=False),
            pl.col("marketing_end_date").str.strptime(
                pl.Date, "%Y%m%d", strict=False)
        ])
        .unique(subset=["product_ndc"])  # One row per product
    )

    return df


def run_pipeline():
    raw_data = fetch_ndc_directory()
    if not raw_data:
        return

    df = process_ndc_directory(raw_data)

    if df is not None:
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        output_path = os.path.join(
            PROCESSED_DATA_PATH, "ndc_directory.parquet")
        df.write_parquet(output_path)

        print(f"   ✅ SUCCESS! Mapped {df.height} drugs.")
        print(f"   💾 Saved to: {output_path}")


if __name__ == "__main__":
    run_pipeline()
