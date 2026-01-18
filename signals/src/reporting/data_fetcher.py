import polars as pl
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Robust Path Finding
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
PROCESSED_PATH = os.path.join(project_root, "data/processed")
NADAC_HISTORY_PATH = os.path.join(PROCESSED_PATH, "nadac_history.parquet")


def get_drug_history(ndc11: str) -> pd.DataFrame:
    """
    Retrieves the full price history for a given drug NDC.
    Forces NDC matching by zero-padding input and database column.
    """
    try:
        if not os.path.exists(NADAC_HISTORY_PATH):
            return pd.DataFrame(columns=['date', 'price'])

        # Defensive: Force target to be 11-digit string
        target_ndc = str(ndc11).zfill(11).strip()

        # Lazy scan for performance
        lf = pl.scan_parquet(NADAC_HISTORY_PATH)

        drug_history = (
            lf
            .select(["ndc11", "effective_date", "price_per_unit"])
            # Defensive: Force DB column to be 11-digit string
            .with_columns(pl.col("ndc11").cast(pl.Utf8).str.strip_chars().str.zfill(11))
            .filter(pl.col("ndc11") == target_ndc)
            .select([
                pl.col("effective_date").alias("date"),
                pl.col("price_per_unit").alias("price")
            ])
            .sort("date")
            .collect()
        )

        if drug_history.height == 0:
            return pd.DataFrame(columns=['date', 'price'])

        return drug_history.to_pandas()

    except Exception as e:
        print(f"❌ Error fetching drug history: {e}")
        return pd.DataFrame(columns=['date', 'price'])


def get_mock_forecast(history_df: pd.DataFrame, risk_score: float) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=['date', 'price', 'lower_bound', 'upper_bound'])

    last_row = history_df.iloc[-1]
    last_price = float(last_row['price'])
    last_date = last_row['date']

    if risk_score > 0.8:
        final_price = last_price * 1.15
    elif risk_score > 0.5:
        final_price = last_price * 1.05
    else:
        final_price = last_price

    future_dates = [last_date + timedelta(weeks=i) for i in range(1, 5)]
    forecast_prices = np.linspace(
        start=last_price, stop=final_price, num=5)[1:]

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'price': forecast_prices
    })

    uncertainty = 0.10 if risk_score > 0.8 else 0.02
    forecast_df['lower_bound'] = forecast_df['price'] * (1 - uncertainty)
    forecast_df['upper_bound'] = forecast_df['price'] * (1 + uncertainty)

    return forecast_df
