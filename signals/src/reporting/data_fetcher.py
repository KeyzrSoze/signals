import polars as pl
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_PATH = 'data/processed'
NADAC_HISTORY_PATH = os.path.join(PROCESSED_PATH, 'nadac_history.parquet')


def get_drug_history(ndc11: str) -> pd.DataFrame:
    """
    Retrieves the full price history for a given drug NDC.

    Args:
        ndc11 (str): The 11-digit NDC of the drug to look up.

    Returns:
        pd.DataFrame: A Pandas DataFrame with 'date' and 'price' columns,
                      sorted by date. Returns an empty DataFrame if the
                      NDC is not found or an error occurs.
    """
    try:
        history_df = pl.read_parquet(NADAC_HISTORY_PATH)
        
        drug_history = (
            history_df
            .filter(pl.col("ndc11") == ndc11)
            .select(
                pl.col("effective_date").alias("date"),
                pl.col("price_per_unit").alias("price")
            )
            .sort("date")
        )
        
        return drug_history.to_pandas()
        
    except Exception as e:
        print(f"Error fetching drug history for NDC {ndc11}: {e}")
        return pd.DataFrame(columns=['date', 'price'])


def get_mock_forecast(history_df: pd.DataFrame, risk_score: float) -> pd.DataFrame:
    """
    Generates a simple, mock forecast based on the last known price and a risk score.

    This function simulates a TFT model output for demonstration purposes.

    Args:
        history_df (pd.DataFrame): The historical price data for a drug.
        risk_score (float): The predicted risk score (0.0 to 1.0) for the drug.

    Returns:
        pd.DataFrame: A DataFrame containing the 4-week mock forecast with
                      'date', 'price', 'lower_bound', and 'upper_bound' columns.
    """
    if history_df.empty:
        return pd.DataFrame(columns=['date', 'price', 'lower_bound', 'upper_bound'])

    last_row = history_df.iloc[-1]
    last_price = last_row['price']
    last_date = last_row['date']

    # Project the final price based on the risk score
    if risk_score > 0.8:
        # High risk projects a 15% price increase over 4 weeks
        final_price = last_price * 1.15
    else:
        # Low or moderate risk projects a flat trend line
        final_price = last_price

    # Create the date range and price trend for the next 4 weeks
    # We create 5 points to get a smooth line from the last known price, then drop the first point.
    future_dates = pd.to_datetime([last_date + timedelta(weeks=i) for i in range(1, 5)])
    forecast_prices = np.linspace(start=last_price, stop=final_price, num=5)[1:]

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'price': forecast_prices
    })

    # Add simple confidence bounds
    forecast_df['lower_bound'] = forecast_df['price'] * 0.95
    forecast_df['upper_bound'] = forecast_df['price'] * 1.05

    return forecast_df


if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Running Data Fetcher examples ---")
    
    # Example NDC known to be in the dataset
    example_ndc = "00078013301" 
    
    # 1. Fetch history
    print(f"\n1. Fetching price history for NDC: {example_ndc}...")
    history = get_drug_history(example_ndc)
    if not history.empty:
        print("   ✅ Found history:")
        print(history.tail())
    else:
        print("   ❌ No history found for this NDC.")

    # 2. Generate a mock forecast for a high-risk scenario
    print("\n2. Generating mock forecast for a HIGH risk scenario (Score > 0.8)...")
    high_risk_forecast = get_mock_forecast(history, risk_score=0.9)
    print(high_risk_forecast)
    
    # 3. Generate a mock forecast for a low-risk scenario
    print("\n3. Generating mock forecast for a LOW risk scenario (Score < 0.5)...")
    low_risk_forecast = get_mock_forecast(history, risk_score=0.3)
    print(low_risk_forecast)

    print("\n--- Examples finished ---")
