import polars as pl
import xgboost as xgb
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
import base64

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_PATH = "data/processed"
MODELS_PATH = "src/models/artifacts"
OUTPUT_PATH = "data/outputs"
TEMPLATE_DIR = "src/reporting/templates"
IMAGE_DIR = "reports/"
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

        # Check for data staleness
        date_difference = datetime.now().date() - latest_date
        is_stale = date_difference > timedelta(days=7)
        stale_warning_msg = ''
        if is_stale:
            stale_warning_msg = 'WARNING: DATA IS OUT OF DATE. DO NOT TRADE.'
            print(
                f"   \n   ⚠️  {stale_warning_msg} (Data is {date_difference.days} days old)   ⚠️\n")

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
            pl.col("risk_score").round(3).alias("score"),
            pl.col("drug_description").alias("drug_name"),
            pl.col("manufacturer"),
            # Assuming risk_type is in the report, else will be null
            pl.lit("Price Instability").alias("risk_type"),
            pl.col("price_per_unit").alias("current_price"),
            pl.col("price_velocity_4w").round(3).alias("momentum"),
            pl.col("market_hhi").round(2).alias("monopoly_index"),
        ])
    )

    # 5. Generate and Save HTML Report
    print("   📄 Generating HTML Report...")
    try:
        top_risks_data = final_report.head(15).to_dicts()

        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template("risk_report.html")

        image_path = os.path.join(IMAGE_DIR, "tft_forecast_plot.png")
        embedded_image_base64 = None
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                embedded_image_base64 = base64.b64encode(
                    img_file.read()).decode('utf-8')
        else:
            print(
                f"   ⚠️ Warning: Image not found at '{image_path}', report will not include chart.")

        html_output = template.render(
            report_date=datetime.now().strftime("%B %d, %Y at %H:%M:%S UTC"),
            data_is_stale=is_stale,
            top_risks=top_risks_data,
            embedded_image_base64=embedded_image_base64
        )

        html_filename = f"Weekly_Risk_Brief_{latest_date}.html"
        html_save_path = os.path.join(OUTPUT_PATH, html_filename)
        with open(html_save_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"   ✅ HTML Report Generated and saved to {html_save_path}")

    except Exception as e:
        print(f"   ❌ Failed to generate HTML report: {e}")

    # 6. Save Raw CSV Report (as backup)
    print("   💾 Saving backup CSV report...")
    filename = f"Risk_Report_{latest_date}.csv"
    save_path = os.path.join(OUTPUT_PATH, filename)
    final_report.write_csv(save_path)

    print(
        f"\n   ✅ Process Complete: {final_report.height} High-Risk Drugs Found")
    print(f"   Top 5 Risks:")
    print(final_report.head(5).select(
        ["drug_name", "score", "momentum"]))
    print(f"\n   💾 CSV backup saved to: {save_path}")


if __name__ == "__main__":
    generate_risk_report()
