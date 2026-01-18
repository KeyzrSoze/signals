import polars as pl
import xgboost as xgb
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
import base64
from signals.src.reporting.data_fetcher import get_drug_history, get_mock_forecast
from signals.src.reporting.interactive_plot import generate_interactive_forecast

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
    try:
        print("   📂 Loading Market Data...")
        features_df = pl.read_parquet(os.path.join(
            PROCESSED_PATH, "weekly_features.parquet"))
        latest_date = features_df["effective_date"].max()
        print(f"   📅 Reporting Date: {latest_date}")
        date_difference = datetime.now().date() - latest_date
        is_stale = date_difference > timedelta(days=7)
        if is_stale:
            print(f"   \n   ⚠️  WARNING: DATA IS OUT OF DATE. (Data is {date_difference.days} days old)   ⚠️\n")
        current_market = features_df.filter(pl.col("effective_date") == latest_date)
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # 2. Load Prediction Model
    print("   🧠 Loading Prediction Model...")
    try:
        model_path = os.path.join(MODELS_PATH, "spike_predictor_v2.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # 3. Predict the Future
    feature_cols = [
        "is_shortage", "weeks_in_shortage", "price_velocity_4w",
        "price_volatility_12w", "market_hhi", "num_competitors"
    ]
    X_live = current_market.select(feature_cols).to_pandas()
    probs = model.predict_proba(X_live)[:, 1]
    report = current_market.with_columns(pl.Series(name="risk_score", values=probs))

    # 4. Format Report
    print("   💄 Formatting for Client...")
    entity_map = pl.read_parquet(os.path.join(PROCESSED_PATH, "ndc_entity_map.parquet"))
    nadac_desc = pl.read_parquet(os.path.join(PROCESSED_PATH, "nadac_history.parquet")).select(["ndc11", "drug_description"]).unique(subset=["ndc11"])

    final_report = (
        report
        .join(entity_map, on="ndc11", how="left")
        .join(nadac_desc, on="ndc11", how="left")
        .filter(pl.col("risk_score") > 0.50)
        .sort("risk_score", descending=True)
        .select([
            pl.col("ndc11"), # Keep NDC for fetching history
            pl.col("risk_score").round(3).alias("score"),
            pl.col("drug_description").alias("drug_name"),
            pl.col("manufacturer"),
            pl.lit("Price Instability").alias("risk_type"),
        ])
    )

    # 5. Generate and Save HTML Report
    print("   📄 Generating HTML Report...")
    try:
        top_risks_data = final_report.head(15).to_dicts()

        # --- Generate Interactive Chart for Top Risk ---
        chart_html = None
        if not final_report.is_empty():
            top_risk = final_report.row(0, named=True)
            print(f"   📈 Generating forecast plot for top risk: {top_risk['drug_name']}")
            history_df = get_drug_history(top_risk['ndc11'])
            forecast_df = get_mock_forecast(history_df, top_risk['score'])
            chart_html = generate_interactive_forecast(history_df, forecast_df, top_risk['drug_name'])
        
        # --- Render Jinja2 Template ---
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template("risk_report.html")
        html_output = template.render(
            report_date=datetime.now().strftime("%B %d, %Y at %H:%M:%S UTC"),
            data_is_stale=is_stale,
            top_risks=top_risks_data,
            interactive_chart=chart_html
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
    print(f"   💾 CSV backup saved to: {save_path}")
    
    print("\n--- Process Complete ---")


if __name__ == "__main__":
    generate_risk_report()
