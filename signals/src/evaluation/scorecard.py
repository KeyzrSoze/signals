import polars as pl
import os
import pandas
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import xgboost as xgb
import sys

# ==========================================
# 🛡️ PATH CORRECTION (GPS Locator)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# CONFIGURATION
# ==========================================
REGISTRY_PATH = os.path.join(
    project_root, 'data/outputs/prediction_registry.parquet')
FEATURES_PATH = os.path.join(
    project_root, 'data/processed/weekly_features.parquet')
NADAC_HISTORY_PATH = os.path.join(
    project_root, 'data/processed/nadac_history.parquet')
MODEL_PATH = os.path.join(
    project_root, 'src/models/artifacts/spike_predictor_v2.pkl')
REPORTS_DIR = os.path.join(project_root, 'reports')

os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def initialize_registry() -> pl.DataFrame:
    if os.path.exists(REGISTRY_PATH):
        print(f"✅ Loading registry from '{REGISTRY_PATH}'.")
        return pl.read_parquet(REGISTRY_PATH)
    else:
        print(f"🛠️ Initializing new registry at '{REGISTRY_PATH}'...")
        schema = {
            "prediction_id": pl.Utf8, "prediction_date": pl.Date, "target_date": pl.Date,
            "ndc11": pl.Utf8, "drug_name": pl.Utf8, "start_price": pl.Float64,
            "predicted_risk_score": pl.Float64, "actual_price": pl.Float64,
            "price_change_pct": pl.Float64, "status": pl.Utf8
        }
        df = pl.DataFrame(schema=schema)
        df.write_parquet(REGISTRY_PATH)
        print("   ✅ Saved new empty registry.")
        return df


def log_new_predictions(registry_df):
    print("\n🔮 Fortuneteller: Logging new predictions...")
    if not os.path.exists(FEATURES_PATH):
        print("   ⚠️ No features file found. Skipping.")
        return registry_df

    features_df = pl.read_parquet(FEATURES_PATH)
    latest_date = features_df["effective_date"].max()
    current_preds = features_df.filter(pl.col("effective_date") == latest_date)

    # 1. Generate scores if missing
    if "risk_score" not in features_df.columns:
        print("   ⚠️ Risk score not found. Generating on the fly...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        feature_cols = ["is_shortage", "weeks_in_shortage", "price_velocity_4w",
                        "price_volatility_12w", "market_hhi", "num_competitors"]

        # Ensure cols exist
        available_cols = [
            c for c in feature_cols if c in current_preds.columns]
        X = current_preds.select(available_cols).to_pandas()
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0

        scores = model.predict_proba(X)[:, 1]
        current_preds = current_preds.with_columns(
            pl.Series("risk_score", scores))

    # 2. FIX: Join with NADAC History to get 'drug_description'
    # The features file has 'ingredient' but usually not the full 'drug_description'
    if "drug_description" not in current_preds.columns:
        print("   📖 Looking up drug names from NADAC history...")
        if os.path.exists(NADAC_HISTORY_PATH):
            names_df = pl.read_parquet(NADAC_HISTORY_PATH).select(
                ["ndc11", "drug_description"]).unique(subset=["ndc11"])
            current_preds = current_preds.join(
                names_df, on="ndc11", how="left")

            # Fill missing names with Ingredient if name lookup failed
            current_preds = current_preds.with_columns(
                pl.col("drug_description").fill_null(pl.col("ingredient"))
            )
        else:
            print("   ⚠️ NADAC history not found. Using 'ingredient' as fallback name.")
            current_preds = current_preds.with_columns(
                pl.col("ingredient").alias("drug_description")
            )

    # 3. Select and Format
    new_entries = (
        current_preds
        .filter(pl.col("risk_score") > 0.5)
        .select([
            pl.col("effective_date").alias("prediction_date"),
            pl.col("ndc11"),
            pl.col("drug_description").alias("drug_name"),
            pl.col("price_per_unit").alias("start_price"),
            pl.col("risk_score").alias("predicted_risk_score")
        ])
        .with_columns([
            (pl.col("prediction_date") + timedelta(weeks=4)).alias("target_date"),
            pl.lit("PENDING").alias("status"),
            pl.lit(None).cast(pl.Float64).alias("actual_price"),
            pl.lit(None).cast(pl.Float64).alias("price_change_pct"),
        ])
        .with_columns([
            (pl.col("prediction_date").cast(pl.Utf8) +
             "_" + pl.col("ndc11")).alias("prediction_id")
        ])
    )

    # Filter duplicates
    if not registry_df.is_empty():
        existing_ids = registry_df["prediction_id"].unique()
        new_entries = new_entries.filter(
            ~pl.col("prediction_id").is_in(existing_ids))

    print(f"   ✅ Logging {new_entries.height} new predictions.")
    return pl.concat([registry_df, new_entries], how="vertical_relaxed")


def reconcile_pending(registry_df):
    print("\n🕵️ Auditor: Reconciling past predictions...")
    if not os.path.exists(NADAC_HISTORY_PATH):
        return registry_df

    today = datetime.now().date()
    pending = registry_df.filter(
        (pl.col("status") == "PENDING") &
        (pl.col("target_date") <= today)
    )

    if pending.is_empty():
        print("   ✅ No pending predictions are due for reconciliation.")
        return registry_df

    print(f"   Checking {pending.height} records against NADAC...")
    history = pl.read_parquet(NADAC_HISTORY_PATH).select(
        ["ndc11", "effective_date", "price_per_unit"])

    # Left Join to get actual prices
    annotated = (
        registry_df
        .join(history, left_on=["ndc11", "target_date"], right_on=["ndc11", "effective_date"], how="left")
        .with_columns([
            pl.when((pl.col("status") == "PENDING") & (
                pl.col("price_per_unit").is_not_null()))
            .then(pl.col("price_per_unit"))
            .otherwise(pl.col("actual_price"))
            .alias("actual_price")
        ])
        .with_columns([
            pl.when((pl.col("status") == "PENDING") & (
                pl.col("actual_price").is_not_null()))
            .then((pl.col("actual_price") - pl.col("start_price")) / pl.col("start_price"))
            .otherwise(pl.col("price_change_pct"))
            .alias("price_change_pct"),

            pl.when((pl.col("status") == "PENDING") & (
                pl.col("price_per_unit").is_not_null()))
            .then(pl.lit("RESOLVED"))
            .otherwise(pl.col("status"))
            .alias("status")
        ])
        .drop("price_per_unit")
    )

    return annotated


def generate_accuracy_plot(registry_df):
    resolved = registry_df.filter(pl.col("status") == "RESOLVED")
    if resolved.is_empty():
        return

    resolved = resolved.with_columns([
        (pl.col("price_change_pct") > 0.05).alias("is_correct")
    ])

    accuracy_over_time = (
        resolved.group_by("prediction_date")
        .agg(pl.col("is_correct").mean().alias("accuracy"))
        .sort("prediction_date")
    ).to_pandas()

    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_over_time["prediction_date"],
             accuracy_over_time["accuracy"], marker='o')
    plt.title("Model Trust Score (Accuracy over Time)")
    plt.ylabel("Accuracy %")
    plt.ylim(0, 1.1)
    plt.grid(True)

    output_path = os.path.join(REPORTS_DIR, "model_trust_score.png")
    plt.savefig(output_path)
    print(f"   ✅ Model accuracy plot saved to '{output_path}'")


if __name__ == '__main__':
    print("--- Running Prediction Registry ETL ---")
    registry = initialize_registry()
    registry = reconcile_pending(registry)
    registry = log_new_predictions(registry)
    registry.write_parquet(REGISTRY_PATH)
    generate_accuracy_plot(registry)
    print("\n   ✅ Process Complete.")
