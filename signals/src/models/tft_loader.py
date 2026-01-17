import os
import polars as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "signals/data/processed/weekly_features.parquet"
MIN_SERIES_LENGTH = 20
MAX_ENCODER_LENGTH = 36  # How much history to use for encoding
MAX_PREDICTION_LENGTH = 12 # How many weeks to predict into the future
BATCH_SIZE = 128

def create_tft_dataloaders(batch_size: int = BATCH_SIZE) -> tuple:
    """
    Loads and prepares the weekly features data for training a Temporal Fusion Transformer.

    Args:
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        A tuple containing: (train_dataloader, val_dataloader, training_dataset)
    """
    logging.info(f"Loading data from '{DATA_PATH}'...")
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file not found at {DATA_PATH}. Please generate features first.")
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pl.read_parquet(DATA_PATH)
    logging.info(f"Loaded {len(df)} records.")

    # --- 2. Data Cleaning & Feature Engineering ---
    logging.info("Starting data cleaning and preparation...")

    # Filter out time series with insufficient history
    series_lengths = df.group_by("ndc11").count().filter(pl.col("count") >= MIN_SERIES_LENGTH)
    df = df.join(series_lengths.select("ndc11"), on="ndc11", how="inner")
    logging.info(f"Filtered to {len(df)} records with at least {MIN_SERIES_LENGTH} weeks of history.")

    # Create a continuous time index for each group
    df = df.sort(["ndc11", "effective_date"])
    df = df.with_columns(
        pl.arange(0, pl.count()).over("ndc11").alias("time_idx")
    )
    
    # Fill nulls and cast types for PyTorch
    # TimeSeriesDataSet requires all real-valued features to be float
    feature_cols = [
        "price_per_unit", "price_velocity_4w", "price_volatility_12w",
        "market_hhi", "num_competitors", "is_shortage", "weeks_in_shortage",
        "manufacturer_risk_score"
    ]
    for col in feature_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0).cast(pl.Float32))
    
    # Cast categorical columns to Polars Categorical type
    df = df.with_columns([
        pl.col("ndc11").cast(pl.Categorical),
        pl.col("manufacturer").cast(pl.Categorical),
        pl.col("ingredient").cast(pl.Categorical),
    ])
    
    logging.info("Data preparation complete.")

    # --- 3. Define TimeSeriesDataSet ---
    logging.info("Creating TimeSeriesDataSet...")

    # Define all available features for the model
    time_varying_unknown_reals = [
        col for col in feature_cols if col in df.columns
    ]
    
    # Create the validation set by holding out the last `max_prediction_length` steps
    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - MAX_PREDICTION_LENGTH

    training_dataset = TimeSeriesDataSet(
        data=df.filter(pl.col("time_idx") <= training_cutoff).to_pandas(),
        time_idx="time_idx",
        target="price_per_unit",
        group_ids=["ndc11"],
        static_categoricals=["manufacturer", "ingredient"],
        time_varying_known_reals=["time_idx"], # time_idx is the only feature known in advance
        time_varying_unknown_reals=time_varying_unknown_reals,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        target_normalizer=GroupNormalizer(groups=["ndc11"], transformation="softplus"),
        # Other normalizers can be added here, e.g., for features
        scalers={col: GroupNormalizer(groups=["ndc11"]) for col in time_varying_unknown_reals if col != "price_per_unit"},
        allow_missing_timesteps=True # Very important for real-world data
    )

    # Create validation dataset from the training dataset
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df.to_pandas(), # The full dataframe is needed to find the validation sequences
        predict=True,
        stop_randomization=True,
    )
    logging.info("TimeSeriesDataSet created successfully.")

    # --- 4. Return PyTorch DataLoaders ---
    logging.info(f"Creating DataLoaders with batch size {batch_size}...")
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    return train_dataloader, val_dataloader, training_dataset

if __name__ == '__main__':
    try:
        train_loader, val_loader, train_ds = create_tft_dataloaders()
        
        logging.info("\n--- Dataloader Information ---")
        logging.info(f"Number of training batches: {len(train_loader)}")
        logging.info(f"Number of validation batches: {len(val_loader)}")
        
        # Inspect a sample batch
        x, y = next(iter(train_loader))
        logging.info("\n--- Sample Batch Shapes ---")
        for key, value in x.items():
            logging.info(f"  Input '{key}': {value.shape}")
        logging.info(f"  Target 'y': {y[0].shape}")
        logging.info(f"  Target weights: {y[1].shape}")
        logging.info("\n✅ Script finished successfully.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
