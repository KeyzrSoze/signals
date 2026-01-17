import polars as pl
import os
import re

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_PATH = "data/processed"
NADAC_PATH = os.path.join(PROCESSED_PATH, "nadac_history.parquet")
EVENTS_PATH = os.path.join(PROCESSED_PATH, "shortage_events.parquet")
MAP_PATH = os.path.join(PROCESSED_PATH, "ndc_entity_map.parquet")
SENTINEL_RISK_PATH = os.path.join(PROCESSED_PATH, "sentinel_risks.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_PATH, "weekly_features.parquet")


def normalize_text(col_expr):
    """Robust text normalization for fuzzy joining."""
    return (
        col_expr
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9\s]", "")
        .str.split(" ")
        .list.get(0)
        .alias("join_key")
    )


def generate_features():
    print("🚀 Starting 'Kitchen Sink' Feature Engineering...")

    # 1. Load Data
    try:
        print("   📂 Loading Datasets...")
        nadac = pl.read_parquet(NADAC_PATH)
        events = pl.read_parquet(EVENTS_PATH)
        entity_map = pl.read_parquet(MAP_PATH)
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # -------------------------------------------------------
    # PHASE 1: MARKET STRUCTURE (Competition & Monopoly)
    # -------------------------------------------------------
    print("   🏗️  Calculating Market Structure (HHI & Competition)...")

    # We need to know: For every week, for every ingredient, WHO is selling?
    # Join NADAC (Price/Date) with Entity Map (Ingredient/Manufacturer)
    market_spine = (
        nadac.select(["effective_date", "ndc11"])
        .join(entity_map.select(["ndc11", "ingredient", "manufacturer"]), on="ndc11", how="inner")
    )

    # Calculate Market Share per Manufacturer per Week
    # Definition: Share = (Count of NDCs for this Mfg) / (Total NDCs for Ingredient)
    market_stats = (
        market_spine
        .group_by(["effective_date", "ingredient", "manufacturer"])
        .agg(pl.count("ndc11").alias("mfg_ndc_count"))
        .with_columns([
            pl.col("mfg_ndc_count").sum().over(
                ["effective_date", "ingredient"]).alias("total_ingredient_ndcs")
        ])
        .with_columns([
            (pl.col("mfg_ndc_count") /
             pl.col("total_ingredient_ndcs")).alias("market_share")
        ])
    )

    # Calculate HHI (Sum of Squared Shares) & Competitor Count
    # HHI range: 0 (Perfect Competition) to 1 (Monopoly)
    competition_features = (
        market_stats
        .group_by(["effective_date", "ingredient"])
        .agg([
            (pl.col("market_share") ** 2).sum().alias("market_hhi"),
            pl.col("manufacturer").n_unique().alias("num_competitors")
        ])
    )

    # -------------------------------------------------------
    # PHASE 2: PRICE DYNAMICS (Velocity & Volatility)
    # -------------------------------------------------------
    print("   📉 Calculating Price Dynamics (Velocity & Volatility)...")

    price_features = (
        nadac
        .sort(["ndc11", "effective_date"])
        .with_columns([
            # 1. Price Momentum (4-Week Velocity)
            # (Current Price - Price 4 Weeks Ago) / Price 4 Weeks Ago
            pl.col("price_per_unit").shift(4).over(
                "ndc11").alias("price_lag_4w"),

            # 2. Volatility (12-Week Coefficient of Variation)
            # StdDev / Mean
            pl.col("price_per_unit").rolling_std(
                window_size=12).over("ndc11").alias("std_12w"),
            pl.col("price_per_unit").rolling_mean(
                window_size=12).over("ndc11").alias("mean_12w")
        ])
        .with_columns([
            ((pl.col("price_per_unit") - pl.col("price_lag_4w")) /
             pl.col("price_lag_4w")).fill_null(0).alias("price_velocity_4w"),
            (pl.col("std_12w") / pl.col("mean_12w")
             ).fill_null(0).alias("price_volatility_12w")
        ])
        .select(["effective_date", "ndc11", "price_per_unit", "price_velocity_4w", "price_volatility_12w"])
    )

    # -------------------------------------------------------
    # PHASE 3: SHORTAGE SIGNALS (The Time Machine)
    # -------------------------------------------------------
    print("   🕰️  Integrating Shortage Events...")

    # Prepare Spine with Join Key
    spine_enhanced = (
        nadac.select(["effective_date", "ndc11"])
        .join(entity_map, on="ndc11", how="left")
        .with_columns(normalize_text(pl.col("ingredient")))
        .sort("effective_date")
    )

    # Prepare Events
    events_normalized = (
        events
        .with_columns(normalize_text(pl.col("generic_name")))
        .sort("event_date")
        .select(["event_date", "join_key", "event_type", "reason"])
    )

    # As-Of Join
    shortage_signals = spine_enhanced.join_asof(
        events_normalized,
        left_on="effective_date",
        right_on="event_date",
        by="join_key",
        strategy="backward"
    ).with_columns([
        pl.when(pl.col("event_type") == "shortage_start").then(
            1).otherwise(0).alias("is_shortage"),
        pl.when(pl.col("event_type") == "shortage_start")
          .then((pl.col("effective_date") - pl.col("event_date")).dt.total_days() / 7)
          .otherwise(0).alias("weeks_in_shortage")
    ]).select(["effective_date", "ndc11", "is_shortage", "weeks_in_shortage"])

    # -------------------------------------------------------
    # PHASE 4: THE GRAND MERGE
    # -------------------------------------------------------
    print("   🔗 Merging All Features...")

    # Base: Price Features (NDCs)
    # Join: Competition (on Ingredient)
    # Join: Shortages (on NDC)

    # We need Ingredient and Manufacturer on the Price table for joins
    master_table = (
        price_features
        .join(entity_map.select(["ndc11", "ingredient", "manufacturer"]), on="ndc11", how="left")
        .join(competition_features, on=["effective_date", "ingredient"], how="left")
        .join(shortage_signals, on=["effective_date", "ndc11"], how="left")
    )
    
    # -------------------------------------------------------
    # PHASE 5: SENTINEL RISK INTEGRATION
    # -------------------------------------------------------
    master_table = integrate_sentinel_risk(master_table)

    # Fill Nulls (e.g., if no competition data found, assume competitive)
    master_table = master_table.with_columns([
        pl.col("market_hhi").fill_null(0),
        pl.col("num_competitors").fill_null(1),
        pl.col("is_shortage").fill_null(0),
        pl.col("weeks_in_shortage").fill_null(0)
    ])

    # -------------------------------------------------------
    # SAVE
    # -------------------------------------------------------
    print(f"   ✅ Complete! Rows: {master_table.height:,}")
    print(f"   💾 Saving to: {OUTPUT_PATH}")
    master_table.write_parquet(OUTPUT_PATH)


def integrate_sentinel_risk(features_df: pl.DataFrame) -> pl.DataFrame:
    """
    Enriches the feature dataframe with manufacturer risk scores from Sentinel data.
    """
    print("   🛡️  Integrating Sentinel Manufacturer Risk...")

    # 1. Load and prepare sentinel risk data
    try:
        sentinel_risks = pl.read_parquet(SENTINEL_RISK_PATH)
    except Exception as e:
        print(f"   ⚠️  Could not load sentinel risk data, skipping. Error: {e}")
        return features_df.with_columns(pl.lit(0).alias("manufacturer_risk_score"))

    # Ensure manufacturer column exists
    if "manufacturer" not in features_df.columns:
        print("   ⚠️  'manufacturer' column not in features_df, skipping Sentinel risk integration.")
        return features_df.with_columns(pl.lit(0).alias("manufacturer_risk_score"))

    # Aggregated risk: max severity per day per manufacturer
    agg_risks = (
        sentinel_risks
        .group_by(["event_date", "manufacturer"])
        .agg(pl.max("severity_score").alias("severity_score"))
        .sort("event_date")
    )

    # 2. Prepare main features dataframe for join
    # The join requires the dataframe to be sorted by the join key
    features_sorted = features_df.sort("effective_date")

    # 3. Point-in-time join
    # For each row in features, find the latest risk event for that manufacturer
    # that occurred on or before the feature's date.
    features_with_risk = features_sorted.join_asof(
        agg_risks,
        left_on="effective_date",
        right_on="event_date",
        by="manufacturer",
        strategy="backward",
        tolerance="90d"  # Look back 90 days
    )

    # 4. Finalize column
    # The join_asof with tolerance will produce nulls if no event is found in the window.
    # We fill nulls with 0, as per requirements.
    final_df = features_with_risk.with_columns(
        pl.col("severity_score").fill_null(0).alias("manufacturer_risk_score")
    ).drop("severity_score") # drop original column after renaming

    print("   ✅ Sentinel risk integration complete.")
    return final_df


if __name__ == "__main__":
    generate_features()
