import polars as pl
import os
import glob
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"


def fetch_and_process_nadac():
    """
    1. Scans data/raw/ for any NADAC csv files.
    2. Enforces strict types (NDC as String).
    3. Stitches multiple files together (e.g. 2024 + 2025 + 2026).
    4. Saves as optimized Parquet.
    """
    print("🚀 Starting NADAC Ingestion (Local File Mode)...")

    # 1. Find all NADAC files in the raw folder
    # We look for any csv file. You can be more specific if needed.
    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))

    if not csv_files:
        print(f"   ❌ No CSV files found in {RAW_DATA_PATH}!")
        print("      -> Please download the NADAC CSV from Medicaid.gov")
        print("      -> Place it in 'data/raw/' and run this again.")
        return

    print(
        f"   📂 Found {len(csv_files)} file(s): {[os.path.basename(f) for f in csv_files]}")

    # 2. Lazy Load & Schema Enforcement
    # scanning multiple files at once is a superpower of Polars
    try:
        nadac_df = pl.scan_csv(
            os.path.join(RAW_DATA_PATH, "*.csv"),  # Use wildcard to load all
            schema_overrides={
                "NDC": pl.String,                # CRITICAL: Keep leading zeros
                "NADAC_Per_Unit": pl.Float64,
                "Effective_Date": pl.String,     # Read as string first
                "Classification_for_Rate_Setting": pl.String
            },
            ignore_errors=True  # Skip malformed rows if any
        )
    except Exception as e:
        print(f"   ❌ Error reading CSVs: {e}")
        return

    # 3. Processing Pipeline
    print("   🧹 Cleaning, merging, and formatting...")

    processed_df = (
        nadac_df
        .select([
            # Normalize Date
            pl.col("Effective_Date")
              .str.strptime(pl.Date, "%m/%d/%Y", strict=False)
              .alias("effective_date"),

            # Normalize NDC (Ensure 11 digits)
            pl.col("NDC")
              .str.replace_all("-", "")
              .str.zfill(11)
              .alias("ndc11"),

            # Rename columns
            pl.col("NADAC_Per_Unit").alias("price_per_unit"),
            pl.col("NDC_Description").alias("drug_description"),
            pl.col("Classification_for_Rate_Setting").alias("classification")
        ])
        .filter(
            # Filter out bad data
            (pl.col("price_per_unit") > 0) &
            (pl.col("effective_date").is_not_null())
        )
        # Deduplicate if files overlap
        .unique(subset=["effective_date", "ndc11"])
        .sort(["effective_date", "ndc11"])
        .collect()  # Execute
    )

    # 4. Validation
    row_count = processed_df.height
    unique_ndcs = processed_df["ndc11"].n_unique()
    min_date = processed_df["effective_date"].min()
    max_date = processed_df["effective_date"].max()

    print(f"   ✅ Data Loaded: {row_count:,} rows")
    print(f"   ✅ Date Range: {min_date} to {max_date}")
    print(f"   ✅ Unique NDCs: {unique_ndcs:,}")

    # 5. Save Artifact
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, "nadac_history.parquet")

    processed_df.write_parquet(output_path)
    print(f"   💾 Saved optimized file to: {output_path}")


if __name__ == "__main__":
    fetch_and_process_nadac()
