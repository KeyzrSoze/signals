import os
import sys
import torch
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder, TorchNormalizer, EncoderNormalizer

# --- FIX: Register Safe Globals for PyTorch 2.6+ ---
torch.serialization.add_safe_globals([
    GroupNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
    EncoderNormalizer,
    pd.DataFrame,
    pd.Series,
    pd.Index,
    pd.RangeIndex,
    pd.core.indexes.base.Index,
    pd.core.indexes.range.RangeIndex,
    pd.core.internals.managers.BlockManager
])

# --- FIX: Add the project root to the python path ---
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# Suppress extensive PyTorch Forecasting warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from src.models.tft_loader import create_tft_dataloaders
except ImportError:
    try:
        from tft_loader import create_tft_dataloaders
    except ImportError as e:
        logging.error(f"Import error: {e}")
        logging.error("Make sure you are running this from the project root.")
        exit(1)

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_PATH = os.path.join(BASE_DIR, "src/models/artifacts/tft_model.ckpt")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FORECAST_PLOT_PATH = os.path.join(REPORTS_DIR, "tft_forecast_plot.png")


def main():
    logging.info("🚀 Starting TFT model explanation process...")

    # 1. Load Data
    try:
        _, val_loader, train_dataset = create_tft_dataloaders()
    except Exception as e:
        logging.error(f"❌ Data Load Error: {e}")
        return

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        logging.error(
            f"❌ Model checkpoint not found at '{MODEL_PATH}'. Please train the model first.")
        return

    logging.info(f"Loading model from '{MODEL_PATH}'...")
    try:
        # Load on CPU to avoid GPU memory issues during reporting
        model = TemporalFusionTransformer.load_from_checkpoint(
            MODEL_PATH, map_location="cpu")
    except Exception as e:
        logging.error(f"❌ Model Load Error: {e}")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 3. Interpretability: Variable Importance
    logging.info("Calculating and plotting variable importance...")

    try:
        raw_predictions = model.predict(val_loader, mode="raw", return_x=True)
        interpretation = model.interpret_output(
            raw_predictions.output, reduction="sum")

        # FIX: Loop through the dictionary of figures
        # 'figs' is a dict like {'attention': Figure, 'static_variables': Figure, ...}
        figs = model.plot_interpretation(interpretation)

        for name, fig in figs.items():
            save_path = os.path.join(
                REPORTS_DIR, f"tft_interpretation_{name}.png")
            fig.savefig(save_path, dpi=300)
            logging.info(f"✅ Saved importance plot ({name}) to '{save_path}'")
            plt.close(fig)

    except Exception as e:
        logging.warning(f"⚠️ Could not generate importance plot: {e}")

    # 4. Forecasting: Plot a specific prediction
    logging.info("Finding a high-risk example and plotting forecast...")

    try:
        best_example_idx = 0
        fig, ax = plt.subplots(figsize=(12, 7))
        model.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=best_example_idx,
            add_loss_to_title=True,
            ax=ax
        )
        ax.set_title(f"TFT Forecast vs Actuals")
        fig.tight_layout()
        fig.savefig(FORECAST_PLOT_PATH, dpi=300)
        logging.info(f"✅ Saved forecast plot to '{FORECAST_PLOT_PATH}'")
        plt.close(fig)
    except Exception as e:
        logging.warning(f"⚠️ Could not generate forecast plot: {e}")

    logging.info("✅ Explanation process complete.")


if __name__ == '__main__':
    main()
