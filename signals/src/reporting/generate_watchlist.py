import polars as pl
import xgboost as xgb
import os
import pickle
import pandas as pd
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_PATH = "data/processed"
MODELS_PATH = "src/models/artifacts"
OUTPUT_PATH = "data/outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def generate_risk_report():
    print("🚀 Generating Weekly Risk Report...")

    # 1. Load the "Latest State" of the Market
    # We load the features file, but we only care about the MOST RECENT date.
    try:
        print("   📂 Loading Market Data...")
        features_df = pl.read_parquet(os.path.join(
            PROCESSED_PATH, "weekly_features.parquet"))

        # Get the latest date
        latest_date = features_df["effective_date"].max()
        print(f"   📅 Reporting Date: {latest_date}")

        # Filter for ONLY the latest week (The "Live" Data)
        current_market = features_df.filter(
            pl.col("effective_date") == latest_date)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # 2. Load the AI Brain
    print("   🧠 Loading Prediction Model...")
    model_path = os.path.join(MODELS_PATH, "spike_predictor_v2.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 3. Predict the Future
    # We need to prep the data exactly like we did for training
    feature_cols = [
        "is_shortage",
        "weeks_in_shortage",
        "price_velocity_4w",
        "price_volatility_12w",
        "market_hhi",
        "num_competitors"
    ]

    X_live = current_market.select(feature_cols).to_pandas()

    # Get Probabilities (Risk Score)
    # class 1 is the "Spike" class
    probs = model.predict_proba(X_live)[:, 1]

    # Attach Risk Scores back to Polars DataFrame
    report = current_market.with_columns([
        pl.Series(name="risk_score", values=probs)
    ])

    # 4. Make it "Client Ready" (Add Human Names)
    # The features file has codes, but clients want names.
    # We fetch these from the Entity Map and NADAC history.
    print("   💄 Formatting for Client...")

    entity_map = pl.read_parquet(os.path.join(
        PROCESSED_PATH, "ndc_entity_map.parquet"))
    nadac_desc = (
        pl.read_parquet(os.path.join(PROCESSED_PATH, "nadac_history.parquet"))
        .select(["ndc11", "drug_description"])
        .unique(subset=["ndc11"])
    )

    final_report = (
        report
        .join(entity_map, on="ndc11", how="left")
        .join(nadac_desc, on="ndc11", how="left")
        .filter(pl.col("risk_score") > 0.50)  # Only show High Risk (>50%)
        .sort("risk_score", descending=True)
        .select([
            pl.col("effective_date"),
            pl.col("risk_score").round(3),
            pl.col("drug_description"),
            pl.col("manufacturer"),
            pl.col("ingredient"),
            pl.col("price_per_unit").alias("current_price"),
            pl.col("price_velocity_4w").round(3).alias("momentum"),
            pl.col("market_hhi").round(2).alias("monopoly_index"),
            pl.col("is_shortage"),
            pl.col("ndc11")
        ])
    )

    # 5. Save Report
    filename = f"Risk_Report_{latest_date}.csv"
    save_path = os.path.join(OUTPUT_PATH, filename)
    final_report.write_csv(save_path)

    print(
        f"   ✅ Report Generated: {final_report.height} High-Risk Drugs Found")
    print(f"   Top 5 Risks:")
    print(final_report.head(5).select(
        ["drug_description", "risk_score", "momentum"]))
    print(f"\n   💾 Saved to: {save_path}")


if __name__ == "__main__":
    generate_risk_report()
