import polars as pl
import os
import pandas
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import xgboost as xgb


# ==========================================
# CONFIGURATION
# ==========================================
REGISTRY_PATH = 'data/outputs/prediction_registry.parquet'
FEATURES_PATH = 'data/processed/weekly_features.parquet'
NADAC_HISTORY_PATH = 'data/processed/nadac_history.parquet'
MODEL_PATH = 'src/models/artifacts/spike_predictor_v2.pkl'
REPORTS_DIR = 'reports'


def initialize_registry() -> pl.DataFrame:
    """
    Initializes the prediction registry.
    Checks if the registry file exists. If so, loads it. If not, creates an empty
    DataFrame with the correct schema and saves it.
    """
    if os.path.exists(REGISTRY_PATH):
        print(f"✅ Prediction Registry already exists. Loading from '{REGISTRY_PATH}'.")
        return pl.read_parquet(REGISTRY_PATH)
    else:
        print(f"🛠️ Prediction Registry not found. Initializing new registry at '{REGISTRY_PATH}'...")
        schema = {
            "prediction_id": pl.Utf8, "prediction_date": pl.Date, "target_date": pl.Date,
            "ndc11": pl.Utf8, "drug_name": pl.Utf8, "start_price": pl.Float64,
            "predicted_risk_score": pl.Float64, "actual_price": pl.Float64,
            "price_change_pct": pl.Float64, "status": pl.Utf8,
        }
        df = pl.DataFrame(schema=schema)
        try:
            os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
            df.write_parquet(REGISTRY_PATH)
            print("   ✅ Saved new empty registry.")
        except Exception as e:
            print(f"   ❌ Error: Could not save the new registry file: {e}")
        return df


def log_new_predictions(registry_df: pl.DataFrame) -> pl.DataFrame:
    """
    Loads the latest market data, generates predictions, and logs them to the registry.
    This function acts as the "Fortuneteller", making new predictions to be evaluated later.
    """
    print("\n🔮 Fortuneteller: Logging new predictions...")
    
    # 1. Load latest market data
    latest_market_df = pl.read_parquet(FEATURES_PATH)
    prediction_date = latest_market_df["effective_date"].max()
    new_predictions_df = latest_market_df.filter(pl.col("effective_date") == prediction_date)

    # 2. Generate risk scores if they are missing
    if "risk_score" not in new_predictions_df.columns:
        print("   ⚠️ Risk score not found. Generating on the fly...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        feature_cols = [
            "is_shortage", "weeks_in_shortage", "price_velocity_4w",
            "price_volatility_12w", "market_hhi", "num_competitors"
        ]
        X_live = new_predictions_df.select(feature_cols).to_pandas()
        probs = model.predict_proba(X_live)[:, 1]
        
        new_predictions_df = new_predictions_df.with_columns(pl.Series(name="risk_score", values=probs))

    # 3. Join for human-readable drug names
    nadac_desc = pl.read_parquet(NADAC_HISTORY_PATH).select(["ndc11", "drug_description"]).unique(subset=["ndc11"])
    new_predictions_df = new_predictions_df.join(nadac_desc, on="ndc11", how="left")

    # 4. Transform data to match registry schema
    new_predictions_df = new_predictions_df.select([
        pl.col("effective_date").alias("prediction_date"),
        pl.col("ndc11"),
        pl.col("drug_description").alias("drug_name"),
        pl.col("price_per_unit").alias("start_price"),
        pl.col("risk_score").alias("predicted_risk_score"),
    ]).with_columns([
        (pl.col("prediction_date").dt.offset_by("28d")).alias("target_date"),
        pl.lit("PENDING").alias("status"),
        pl.concat_str([pl.col("effective_date").cast(pl.Utf8), pl.col("ndc11")], separator="-").alias("prediction_id")
    ])

    # 5. Anti-Join to filter out predictions already in the registry
    new_unique_rows = new_predictions_df.join(registry_df, on="prediction_id", how="anti")
    
    if new_unique_rows.is_empty():
        print("   ✅ No new predictions to log. Registry is up to date.")
        return registry_df
    
    print(f"   ✍️ Found {len(new_unique_rows)} new predictions to log.")

    # 6. Stack new predictions onto the existing registry and return
    ordered_cols = registry_df.columns
    return pl.concat([registry_df, new_unique_rows.select(ordered_cols)])


def reconcile_pending(registry_df: pl.DataFrame) -> pl.DataFrame:
    """
    Finds predictions whose target date has passed and attempts to resolve them
    by looking up the actual price from historical data.
    """
    print("\n🕵️ Auditor: Reconciling past predictions...")

    # 1. Filter for predictions that are past their target date and still pending
    pending_to_reconcile = registry_df.filter(
        (pl.col("status") == "PENDING") & (pl.col("target_date") <= datetime.now().date())
    )

    if pending_to_reconcile.is_empty():
        print("   ✅ No pending predictions are due for reconciliation.")
        return registry_df

    print(f"   Auditing {len(pending_to_reconcile)} pending predictions...")

    # 2. Load the source of truth for prices
    nadac_truth = pl.read_parquet(NADAC_HISTORY_PATH).select(
        ["ndc11", "effective_date", "price_per_unit"]
    ).sort("effective_date")

    # 3. Join 'asof' to find the price at or just before the target date
    reconciled_df = pending_to_reconcile.sort("target_date").join_asof(
        nadac_truth,
        left_on="target_date",
        right_on="effective_date",
        by="ndc11"
    )

    # 4. Calculate outcomes and update status where a price was found
    updated_rows = reconciled_df.with_columns(
        pl.col("price_per_unit").alias("actual_price"),
        pl.when(pl.col("price_per_unit").is_not_null())
          .then((pl.col("price_per_unit") - pl.col("start_price")) / pl.col("start_price"))
          .otherwise(None)
          .alias("price_change_pct"),
        pl.when(pl.col("price_per_unit").is_not_null())
          .then(pl.lit("RESOLVED"))
          .otherwise(pl.lit("PENDING")) # Keep as PENDING if no price found yet
          .alias("status")
    ).drop("price_per_unit")

    # 5. Merge the updates back into the main registry
    final_registry = registry_df.update(updated_rows.select(registry_df.columns), on="prediction_id")
    
    resolved_count = updated_rows.filter(pl.col('status') == 'RESOLVED').height
    print(f"   ✅ Reconciliation complete. {resolved_count} predictions have been resolved.")

    return final_registry


def generate_accuracy_plot(registry_df: pl.DataFrame):
    """
    Analyzes the accuracy of resolved predictions and saves a plot.
    """
    print("\n📊 Generating Model Trust Score plot...")
    
    # Use 'RESOLVED' status based on our implementation
    resolved_df = registry_df.filter(pl.col("status") == "RESOLVED")

    if resolved_df.is_empty():
        print("   Not enough resolved predictions to generate an accuracy plot yet.")
        return

    # Define "Spike" conditions and check for correctness
    scored_df = resolved_df.with_columns([
        (pl.col("predicted_risk_score") > 0.5).alias("Predicted_Spike"),
        (pl.col("price_change_pct") > 0.05).alias("Actual_Spike")
    ]).with_columns(
        (pl.col("Predicted_Spike") == pl.col("Actual_Spike")).cast(pl.Int8).alias("Correct")
    )

    # Group by date and calculate daily accuracy
    accuracy_over_time = scored_df.group_by("prediction_date").agg(
        pl.mean("Correct").alias("accuracy")
    ).sort("prediction_date")

    if accuracy_over_time.is_empty():
        print("   No data to plot after aggregation.")
        return

    # Plot the data
    plot_df = accuracy_over_time.to_pandas() # Matplotlib works well with Pandas DataFrames
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(plot_df['prediction_date'], plot_df['accuracy'] * 100, marker='o', linestyle='-', color='b')
    
    # Formatting
    ax.set_title('Model Trust Score: Prediction Accuracy Over Time', fontsize=16, weight='bold')
    ax.set_xlabel('Prediction Date', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    fig.autofmt_xdate()

    # Save the plot
    output_path = os.path.join(REPORTS_DIR, 'model_trust_score.png')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"   ✅ Model accuracy plot saved to '{output_path}'")


if __name__ == '__main__':
    print("--- Running Prediction Registry ETL ---")
    
    # 1. Initialize registry (load or create)
    registry = initialize_registry()
    print(f"   Registry started with {len(registry)} records.")
    
    # 2. Reconcile past predictions first
    registry = reconcile_pending(registry)

    # 3. Log new predictions from the latest data
    registry = log_new_predictions(registry)
    
    # 4. Save the updated registry
    registry.write_parquet(REGISTRY_PATH)
    print(f"\n   ✅ Registry updated. Now contains {len(registry)} records.")

    # 5. Generate the accuracy report
    generate_accuracy_plot(registry)

    # 6. Print final summary
    print("\n--- Scorecard Summary ---")
    open_predictions = registry.filter(pl.col("status") == "PENDING").height
    resolved_predictions = registry.filter(pl.col("status") == "RESOLVED").height
    
    current_accuracy = 0.0
    if resolved_predictions > 0:
        correct_count = registry.filter(
            (pl.col("status") == "RESOLVED") &
            ((pl.col("predicted_risk_score") > 0.5) == (pl.col("price_change_pct") > 0.05))
        ).height
        current_accuracy = (correct_count / resolved_predictions) * 100

    print(f"   - Open Predictions:   {open_predictions}")
    print(f"   - Resolved Predictions: {resolved_predictions}")
    print(f"   - Overall Accuracy:   {current_accuracy:.2f}%")

    print("\n--- Process Complete ---")
