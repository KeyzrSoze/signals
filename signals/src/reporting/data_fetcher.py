import polars as pl
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
# Robust Path Finding (Works from root or src)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
PROCESSED_PATH = os.path.join(project_root, "data/processed")
NADAC_HISTORY_PATH = os.path.join(PROCESSED_PATH, "nadac_history.parquet")


def get_drug_history(ndc11: str) -> pd.DataFrame:
    """
    Retrieves the full price history for a given drug NDC.

    CRITICAL: This function forces the NDC column to be a String with 
    leading zeros (11 digits) to match the input.
    """
    try:
        # 1. Check if file exists
        if not os.path.exists(NADAC_HISTORY_PATH):
            print(f"⚠️  History file not found at {NADAC_HISTORY_PATH}")
            return pd.DataFrame(columns=['date', 'price'])

        # 2. Load Data (Lazy mode for speed, then collect)
        # We only read the columns we need to save memory
        lf = pl.scan_parquet(NADAC_HISTORY_PATH)

        # 3. defensive_cleaning_pipeline
        # We assume the DB might have ints, strings, or messy data.
        # We force it into a standard 11-digit string format.
        target_ndc = str(ndc11).zfill(11).strip()

        drug_history = (
            lf
            .select(["ndc11", "effective_date", "price_per_unit"])
            .with_columns([
                # FORCE NDC to be an 11-char string with leading zeros
                pl.col("ndc11").cast(pl.Utf8).str.strip_chars().str.zfill(11)
            ])
            .filter(pl.col("ndc11") == target_ndc)
            .select([
                pl.col("effective_date").alias("date"),
                pl.col("price_per_unit").alias("price")
            ])
            .sort("date")
            .collect()  # Execute the query
        )

        if drug_history.height == 0:
            # Uncomment for debugging only
            # print(f"   ⚠️  NDC {target_ndc} not found in history.")
            return pd.DataFrame(columns=['date', 'price'])

        return drug_history.to_pandas()

    except Exception as e:
        print(f"❌ Error fetching drug history for NDC {ndc11}: {e}")
        return pd.DataFrame(columns=['date', 'price'])


def get_mock_forecast(history_df: pd.DataFrame, risk_score: float) -> pd.DataFrame:
    """
    Generates a forecast based on risk score.
    """
    if history_df.empty:
        return pd.DataFrame(columns=['date', 'price', 'lower_bound', 'upper_bound'])

    last_row = history_df.iloc[-1]
    last_price = float(last_row['price'])
    last_date = last_row['date']

    # Project the final price based on the risk score
    if risk_score > 0.8:
        # High risk projects a 15% price increase over 4 weeks
        final_price = last_price * 1.15
    elif risk_score > 0.5:
        # Medium risk: 5% increase
        final_price = last_price * 1.05
    else:
        # Low risk: Flat
        final_price = last_price

    # Create the date range (Next 4 weeks)
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, 5)]

    # Linear interpolation
    forecast_prices = np.linspace(
        start=last_price, stop=final_price, num=5)[1:]

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'price': forecast_prices
    })

    # Add Confidence Bounds (The "Tunnel")
    # Higher risk = Wider uncertainty
    uncertainty = 0.10 if risk_score > 0.8 else 0.02

    forecast_df['lower_bound'] = forecast_df['price'] * (1 - uncertainty)
    forecast_df['upper_bound'] = forecast_df['price'] * (1 + uncertainty)

    return forecast_df


if __name__ == '__main__':
    # Simple self-test
    print("--- Testing Data Fetcher ---")
    # Use a dummy NDC that likely doesn't exist just to test the "Empty Return" logic
    test = get_drug_history("00000000000")
    print(f"Result empty? {test.empty}")
