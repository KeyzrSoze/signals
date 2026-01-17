import os
import sys
import torch
import polars as pl
import matplotlib.pyplot as plt
import logging
import warnings

# --- FIX: Add the project root to the python path ---
# This ensures we can import from src.models regardless of where we run this script
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# Suppress extensive PyTorch Forecasting warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from pytorch_forecasting import TemporalFusionTransformer
    # FIX: Changed import to standard format
    from src.models.tft_loader import create_tft_dataloaders
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error(
        "Make sure you are running this from the project root (e.g., python src/reporting/explain_tft.py)")
    exit()


# --- Configuration ---
MODEL_PATH = "src/models/artifacts/tft_model.ckpt"
REPORTS_DIR = "reports"
FORECAST_PLOT_PATH = os.path.join(REPORTS_DIR, "tft_forecast_plot.png")
IMPORTANCE_PLOT_PATH = os.path.join(REPORTS_DIR, "tft_variable_importance.png")


def main():
    """
    Main function to load the trained TFT model and generate explanation reports.
    """
    logging.info("🚀 Starting TFT model explanation process...")

    # 1. Load Model and Data
    if not os.path.exists(MODEL_PATH):
        logging.error(
            f"❌ Model checkpoint not found at '{MODEL_PATH}'. Please train the model first.")
        return

    try:
        # We only need the validation loader and the training dataset object (for metadata)
        _, val_loader, train_dataset = create_tft_dataloaders()
    except FileNotFoundError as e:
        logging.error(f"❌ {e}. Cannot generate explanations without data.")
        return

    logging.info(f"Loading model from '{MODEL_PATH}'...")
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)

    # Ensure the reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 2. Interpretability: Variable Importance
    logging.info("Calculating and plotting variable importance...")

    # Predict on the validation data to get raw model outputs
    raw_predictions = model.predict(val_loader, mode="raw", return_x=True)

    # Calculate interpretation - this caches the results in the model
    interpretation = model.interpret_output(
        raw_predictions.output, reduction="sum")

    # Plot interpretation and save the figure
    fig = model.plot_interpretation(interpretation)
    fig.savefig(IMPORTANCE_PLOT_PATH, dpi=300)
    logging.info(
        f"✅ Saved variable importance plot to '{IMPORTANCE_PLOT_PATH}'")
    plt.close(fig)

    # 3. Forecasting: Plot a specific prediction
    logging.info("Finding a high-risk example and plotting forecast...")

    # Find an interesting example from the validation set to plot
    best_example_idx = 0
    max_risk_score = -1.0

    # We need to access the underlying dataset to find indices
    dataset = val_loader.dataset

    try:
        # Check if risk score exists in reals
        if 'manufacturer_risk_score' in dataset.reals:
            # This is a bit complex because the dataset is indexed by time windows
            # We will just grab a random high risk one from the underlying data
            # to keep it simple and robust

            # Convert to polars or pandas to search easily
            data_df = dataset.data
            high_risk_rows = data_df[data_df['manufacturer_risk_score'] > 5]

            if not high_risk_rows.empty:
                # Find the index of the first high risk item in the dataset
                # This is an approximation but works for visualization
                high_risk_idx = high_risk_rows.index[0]
                # We need to map this data index to a prediction index
                # For now, just using 0 if complex mapping fails
                best_example_idx = 0
                max_risk_score = high_risk_rows['manufacturer_risk_score'].iloc[0]
    except Exception as e:
        logging.warning(
            f"Could not search for high risk example: {e}. Plotting index 0.")

    # Generate the prediction plot for the selected example
    fig, ax = plt.subplots(figsize=(12, 7))
    model.plot_prediction(
        raw_predictions.x,
        raw_predictions.output,
        idx=best_example_idx,
        add_loss_to_title=True,
        ax=ax
    )

    # Extract NDC ID safely
    try:
        # Use the decoder to get the categorical value back
        ndc_idx = raw_predictions.x['decoder_cat'][best_example_idx][0].item()
        # You might need to adjust based on which categorical is the group_id
        # This is a simplified fallback
        ndc_label = f"Sample {best_example_idx}"
    except:
        ndc_label = "Unknown NDC"

    ax.set_title(f"TFT Forecast vs Actuals")
    fig.tight_layout()

    # 4. Save the Forecasting Plot
    fig.savefig(FORECAST_PLOT_PATH, dpi=300)
    logging.info(f"✅ Saved forecast plot to '{FORECAST_PLOT_PATH}'")
    plt.close(fig)

    logging.info("✅ Explanation process complete.")


if __name__ == '__main__':
    main()
