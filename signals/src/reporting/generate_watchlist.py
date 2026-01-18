import polars as pl
import xgboost as xgb
import pickle
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader

# ==========================================
# 🛡️ BULLETPROOF IMPORT BLOCK
# ==========================================
try:
    from src.reporting.data_fetcher import get_drug_history, get_mock_forecast
    from src.reporting.interactive_plot import generate_interactive_forecast
except ImportError:
    try:
        from data_fetcher import get_drug_history, get_mock_forecast
        from interactive_plot import generate_interactive_forecast
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.reporting.data_fetcher import get_drug_history, get_mock_forecast
        from src.reporting.interactive_plot import generate_interactive_forecast
# ==========================================

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "src/models/artifacts")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/outputs")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "src/reporting/templates")

os.makedirs(OUTPUT_PATH, exist_ok=True)


def generate_risk_report():
    print("🚀 Generating Weekly Risk Report...")

    # 1. Load Data
    try:
        features_path = os.path.join(PROCESSED_PATH, "weekly_features.parquet")
        if not os.path.exists(features_path):
            print(f"❌ CRITICAL: File not found at {features_path}")
            return

        features_df = pl.read_parquet(features_path)
        latest_date = features_df["effective_date"].max()
        print(f"   📅 Reporting Date: {latest_date}")

        current_market = features_df.filter(
            pl.col("effective_date") == latest_date)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Load Model
    try:
        model_path = os.path.join(MODELS_PATH, "spike_predictor_v2.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Predict
    feature_cols = [
        "is_shortage", "weeks_in_shortage", "price_velocity_4w",
        "price_volatility_12w", "market_hhi", "num_competitors"
    ]

    available_cols = [c for c in feature_cols if c in current_market.columns]
    X_live = current_market.select(available_cols).to_pandas()

    for col in feature_cols:
        if col not in X_live.columns:
            X_live[col] = 0

    probs = model.predict_proba(X_live)[:, 1]
    report = current_market.with_columns(
        pl.Series(name="risk_score", values=probs))

    # 4. Format Report (RESTORED COLUMNS)
    print("   💄 Formatting for Client...")
    try:
        entity_map = pl.read_parquet(os.path.join(
            PROCESSED_PATH, "ndc_entity_map.parquet"))
        nadac_desc = pl.read_parquet(os.path.join(PROCESSED_PATH, "nadac_history.parquet")).select(
            ["ndc11", "drug_description"]).unique(subset=["ndc11"])

        final_report = (
            report
            .join(entity_map, on="ndc11", how="left")
            .join(nadac_desc, on="ndc11", how="left")
            .filter(pl.col("risk_score") > 0.50)
            .sort("risk_score", descending=True)
            .select([
                pl.col("ndc11"),
                pl.col("risk_score").round(3).alias("score"),
                pl.col("drug_description").alias("drug_name"),
                pl.col("manufacturer"),
                pl.lit("Price Instability").alias("risk_type"),
                # --- COLUMNS RESTORED BELOW ---
                pl.col("price_per_unit").alias("current_price"),
                pl.col("price_velocity_4w").round(3).alias("momentum"),
                pl.col("market_hhi").round(2).alias("monopoly_index")
            ])
        )
    except Exception as e:
        print(f"❌ Error formatting report: {e}")
        return

    # 5. Generate HTML
    try:
        top_risks_data = final_report.head(15).to_dicts()
        chart_html = None

        if not final_report.is_empty():
            top_risk = final_report.row(0, named=True)
            print(
                f"   📈 Generating forecast plot for: {top_risk['drug_name']}")

            # Use string cast for NDC lookups to prevent 'missing chart' errors
            history_df = get_drug_history(str(top_risk['ndc11']))
            forecast_df = get_mock_forecast(history_df, top_risk['score'])

            chart_html = generate_interactive_forecast(
                history_df, forecast_df, top_risk['drug_name'])

        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template("risk_report.html")
        html_output = template.render(
            report_date=datetime.now().strftime("%B %d, %Y"),
            top_risks=top_risks_data,
            interactive_chart=chart_html
        )

        html_save_path = os.path.join(
            OUTPUT_PATH, f"Weekly_Risk_Brief_{latest_date}.html")
        with open(html_save_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"   ✅ HTML Report saved: {html_save_path}")

    except Exception as e:
        print(f"❌ Failed to generate HTML report: {e}")
        import traceback
        traceback.print_exc()

    # 6. Save CSV Backup
    save_path = os.path.join(OUTPUT_PATH, f"Risk_Report_{latest_date}.csv")
    final_report.write_csv(save_path)
    print(f"   💾 CSV backup saved: {save_path}")


if __name__ == "__main__":
    generate_risk_report()
