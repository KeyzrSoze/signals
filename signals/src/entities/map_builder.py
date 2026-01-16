import polars as pl
import os
import re

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_DATA_PATH = "data/processed"
NADAC_FILE = os.path.join(PROCESSED_DATA_PATH, "nadac_history.parquet")
NDC_DIR_FILE = os.path.join(PROCESSED_DATA_PATH, "ndc_directory.parquet")


def build_entity_map():
    print("🚀 Starting Entity Map Construction (Smart Fallback Mode)...")

    # 1. Load Data
    try:
        # We need 'drug_description' from NADAC for the text mining fallback
        nadac_df = pl.read_parquet(NADAC_FILE).select(
            ["ndc11", "drug_description"]).unique()
        fda_df = pl.read_parquet(NDC_DIR_FILE)
    except Exception as e:
        print(f"   ❌ Error loading files: {e}")
        return

    # 2. Prepare Link Keys (9-digit)
    print("   🔗 Generating Link Keys...")

    nadac_map = nadac_df.with_columns([
        pl.col("ndc11").str.slice(0, 9).alias("match_key_9"),
        pl.col("ndc11").str.slice(0, 5).alias(
            "labeler_code_5")  # Extract Labeler (First 5)
    ])

    # Prepare FDA (Normalize "12345-6789" -> "123456789")
    fda_map = fda_df.with_columns([
        pl.col("product_ndc")
          .str.split("-")
          .list.get(0)
          .str.zfill(5)
          .alias("labeler_code"),

        pl.col("product_ndc")
          .str.split("-")
          .list.get(1)
          .str.zfill(4)
          .alias("product_code")
    ]).with_columns(
        (pl.col("labeler_code") + pl.col("product_code")).alias("match_key_9")
    )

    # 3. Create a "Labeler Dictionary"
    # Even if we don't match the specific drug, we can match the Manufacturer
    # using the first 5 digits.
    print("   🧠 Building Labeler Knowledge Base...")
    labeler_dict = (
        fda_map
        .select(["labeler_code", "manufacturer_simple"])
        .unique(subset=["labeler_code"])
        .filter(pl.col("manufacturer_simple").is_not_null())
        .rename({"labeler_code": "labeler_code_5", "manufacturer_simple": "mfg_fallback"})
    )

    # 4. The Great Join
    print("   🌉 Bridging Data...")

    # Step A: Direct Join (The 15% precise matches)
    master = nadac_map.join(
        fda_map,
        on="match_key_9",
        how="left"
    )

    # Step B: Join Labeler Fallback
    master = master.join(
        labeler_dict,
        on="labeler_code_5",
        how="left"
    )

    # 5. The "MacGyver" Logic (Fill Gaps)
    print("   🔧 Applying Smart Gap-Filling...")

    # Helper to clean descriptions (e.g. "AMOXICILLIN 500MG" -> "AMOXICILLIN")
    # We strip numbers and common dosage forms to get the raw ingredient name.

    master = master.with_columns([
        # Fallback 1: Manufacturer
        pl.col("manufacturer_simple")
          .fill_null(pl.col("mfg_fallback"))
          .fill_null("UNKNOWN_MFG")
          .alias("final_manufacturer"),

        # Fallback 2: Ingredient (Regex Magic)
        # We take the first word of the description as a crude but effective ingredient proxy
        pl.col("ingredient_name")
          .fill_null(
              pl.col("drug_description")
                .str.split(" ")  # Split "GABAPENTIN 300MG CAP"
                .list.get(0)    # Take "GABAPENTIN"
        )
        .fill_null("UNKNOWN_INGREDIENT")
        .alias("final_ingredient")
    ])

    # 6. Final Clean & Save
    final_map = master.select([
        pl.col("ndc11"),
        pl.col("drug_description"),
        pl.col("final_manufacturer").alias("manufacturer"),
        pl.col("final_ingredient").alias("ingredient"),
        pl.col("labeler_code_5").alias("labeler_id")
    ])

    # Stats
    total = final_map.height
    with_mfg = final_map.filter(pl.col("manufacturer") != "UNKNOWN_MFG").height
    with_ing = final_map.filter(
        pl.col("ingredient") != "UNKNOWN_INGREDIENT").height

    print(f"   ✅ Map Complete!")
    print(f"      Total NDCs: {total:,}")
    print(
        f"      Manufacturer Coverage: {with_mfg:,} ({(with_mfg/total)*100:.1f}%)")
    print(
        f"      Ingredient Coverage:   {with_ing:,} ({(with_ing/total)*100:.1f}%)")

    output_path = os.path.join(PROCESSED_DATA_PATH, "ndc_entity_map.parquet")
    final_map.write_parquet(output_path)
    print(f"   💾 Saved Enhanced Map to: {output_path}")


if __name__ == "__main__":
    build_entity_map()
