import os
import polars as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def find_data_path():
    """
    Robustly finds the weekly_features.parquet file.
    Searches current dir, parent dir, and project root.
    """
    filename = "weekly_features.parquet"
    candidates = [
        "data/processed/" + filename,
        "signals/data/processed/" + filename,
        "../data/processed/" + filename,
        "../../data/processed/" + filename
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # If not found relative, try finding absolute path from script location
    base_dir = os.path.dirname(os.path.abspath(__file__))  # src/models
    root_dir = os.path.dirname(os.path.dirname(base_dir))  # project root
    abs_path = os.path.join(root_dir, "data/processed", filename)

    if os.path.exists(abs_path):
        return abs_path

    return None


DATA_PATH = find_data_path()
MIN_SERIES_LENGTH = 20
MAX_ENCODER_LENGTH = 36
MAX_PREDICTION_LENGTH = 12
BATCH_SIZE = 128


def create_tft_dataloaders(batch_size: int = BATCH_SIZE) -> tuple:
    if DATA_PATH is None:
        raise FileNotFoundError(
            "Could not find 'weekly_features.parquet' in any expected location.")

    logging.info(f"Loading data from '{DATA_PATH}'...")
    df = pl.read_parquet(DATA_PATH)
    logging.info(f"Loaded {len(df)} records.")

    # Data Cleaning
    logging.info("Starting data cleaning and preparation...")
    series_lengths = df.group_by("ndc11").len().filter(
        pl.col("len") >= MIN_SERIES_LENGTH)

    if series_lengths.is_empty():
        raise ValueError(
            f"No NDCs found with history >= {MIN_SERIES_LENGTH} weeks.")

    df = df.join(series_lengths.select("ndc11"), on="ndc11", how="inner")

    df = df.sort(["ndc11", "effective_date"])
    df = df.with_columns(
        pl.arange(0, pl.len()).over("ndc11").alias("time_idx")
    )

    feature_cols = [
        "price_per_unit", "price_velocity_4w", "price_volatility_12w",
        "market_hhi", "num_competitors", "is_shortage", "weeks_in_shortage",
        "manufacturer_risk_score"
    ]
    for col in feature_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0).cast(pl.Float32))

    df = df.with_columns([
        pl.col("ndc11").cast(pl.Categorical),
        pl.col("manufacturer").cast(pl.Categorical),
        pl.col("ingredient").cast(pl.Categorical),
    ])

    logging.info("Creating TimeSeriesDataSet...")

    time_varying_unknown_reals = [c for c in feature_cols if c in df.columns]
    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - MAX_PREDICTION_LENGTH

    training_dataset = TimeSeriesDataSet(
        data=df.filter(pl.col("time_idx") <= training_cutoff).to_pandas(),
        time_idx="time_idx",
        target="price_per_unit",
        group_ids=["ndc11"],
        static_categoricals=["manufacturer", "ingredient"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=time_varying_unknown_reals,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        target_normalizer=GroupNormalizer(
            groups=["ndc11"], transformation="softplus"),
        scalers={col: GroupNormalizer(
            groups=["ndc11"]) for col in time_varying_unknown_reals if col != "price_per_unit"},
        allow_missing_timesteps=True
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df.to_pandas(),
        predict=True,
        stop_randomization=True,
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0)

    return train_dataloader, val_dataloader, training_dataset


if __name__ == '__main__':
    try:
        train_loader, val_loader, train_ds = create_tft_dataloaders()
        logging.info("✅ Script finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
