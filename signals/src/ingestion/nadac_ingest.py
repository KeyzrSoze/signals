import polars as pl
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"


def normalize_and_load(file_path):
    """
    Reads a CSV, forces all headers to lowercase, applies standard naming,
    and returns a clean LazyFrame.
    """
    # 1. Read (Lazy)
    lf = pl.scan_csv(file_path, infer_schema_length=0)

    # 2. Force Lowercase Headers immediately
    # This solves the "NDC" vs "ndc" issue forever.
    # Proper way to get names in newer Polars
    current_columns = lf.collect_schema().names()
    lower_map = {col: col.lower() for col in current_columns}
    lf = lf.rename(lower_map)

    # 3. Define our "Rosetta Stone" (All keys must be lowercase now)
    rename_map = {
        # Price
        "nadac per unit": "price_per_unit",
        "nadac_per_unit": "price_per_unit",

        # Date
        "effective date": "effective_date",
        "effective_date": "effective_date",

        # NDC
        "ndc": "ndc11",

        # Description
        "ndc description": "drug_description",
        "ndc_description": "drug_description",

        # Classification
        "classification for rate setting": "classification",
        "classification_for_rate_setting": "classification"
    }

    # 4. Apply Renaming
    # We re-fetch columns because we just renamed them to lowercase
    current_columns_lower = [c.lower() for c in current_columns]
    actual_renames = {k: v for k,
                      v in rename_map.items() if k in current_columns_lower}
    lf = lf.rename(actual_renames)

    # 5. Standardize Data Types
    lf = lf.select([
        pl.col("effective_date").str.strptime(
            pl.Date, "%m/%d/%Y", strict=False),

        pl.col("ndc11")
          .str.replace_all("-", "")
          .str.zfill(11),

        pl.col("price_per_unit").cast(pl.Float64),

        pl.col("drug_description"),
        pl.col("classification")
    ])

    return lf


def fetch_and_process_nadac():
    print("🚀 Starting NADAC Ingestion (Case-Insensitive Mode)...")

    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
    if not csv_files:
        print("   ❌ No CSV files found!")
        return

    print(f"   📂 Processing {len(csv_files)} files...")

    lazy_frames = []
    for f in csv_files:
        try:
            lf = normalize_and_load(f)
            lazy_frames.append(lf)
        except Exception as e:
            print(f"      ⚠️ Skipping {os.path.basename(f)}: {e}")

    if not lazy_frames:
        return

    print("   🔗 Merging history...")
    combined_lf = pl.concat(lazy_frames)

    # Final Polish
    final_df = (
        combined_lf
        .filter(pl.col("price_per_unit").is_not_null())
        .unique(subset=["effective_date", "ndc11"])
        .sort(["effective_date", "ndc11"])
        .collect()
    )

    # Validation
    row_count = final_df.height
    min_date = final_df["effective_date"].min()
    max_date = final_df["effective_date"].max()

    print(f"   ✅ SUCCESS! History Range: {min_date} to {max_date}")
    print(f"   ✅ Total Rows: {row_count:,}")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, "nadac_history.parquet")
    final_df.write_parquet(output_path)
    print(f"   💾 Saved to: {output_path}")


if __name__ == "__main__":
    fetch_and_process_nadac()
