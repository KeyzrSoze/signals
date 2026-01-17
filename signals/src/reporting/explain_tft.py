import os
import torch
import polars as pl
import matplotlib.pyplot as plt
import logging
import warnings

# Suppress extensive PyTorch Forecasting warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from pytorch_forecasting import TemporalFusionTransformer
    # Assumes execution from project root
    from signals.src.models.tft_loader import create_tft_dataloaders
except ImportError as e:
    logging.error(f"Import error: {e}. Make sure all dependencies are installed and you are running from the project root.")
    exit()


# --- Configuration ---
MODEL_PATH = "signals/src/models/artifacts/tft_model.ckpt"
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
        logging.error(f"❌ Model checkpoint not found at '{MODEL_PATH}'. Please train the model first.")
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
    interpretation = model.interpret_output(raw_predictions.output, reduction="sum")
    
    # Plot interpretation and save the figure
    fig = model.plot_interpretation(interpretation)
    fig.savefig(IMPORTANCE_PLOT_PATH, dpi=300)
    logging.info(f"✅ Saved variable importance plot to '{IMPORTANCE_PLOT_PATH}'")
    plt.close(fig)


    # 3. Forecasting: Plot a specific prediction
    logging.info("Finding a high-risk example and plotting forecast...")

    # Find an interesting example from the validation set to plot
    # Here, we look for an NDC with a recent, non-zero manufacturer risk score
    best_example_idx = 0
    max_risk_score = -1.0
    try:
        risk_feature_idx = val_loader.dataset.reals.index('manufacturer_risk_score')
        for i, (x, y) in enumerate(iter(val_loader)):
            # Get the risk scores for the last time step in the encoder
            last_encoder_risks = x['encoder_cont'][:, -1, risk_feature_idx]
            batch_max_risk = last_encoder_risks.max().item()
            if batch_max_risk > max_risk_score:
                max_risk_score = batch_max_risk
                # Find the index within the batch and map it back to the dataloader index
                item_idx_in_batch = torch.argmax(last_encoder_risks).item()
                best_example_idx = i * val_loader.batch_size + item_idx_in_batch
            # Stop after checking a few batches to speed things up
            if i > 20 and max_risk_score > 0:
                break
    except ValueError:
        logging.warning("`manufacturer_risk_score` not in features. Plotting first example.")
    
    if max_risk_score > 0:
        logging.info(f"Found example with risk score {max_risk_score:.2f}. Plotting index {best_example_idx}.")
    else:
        logging.info("No examples with risk score > 0 found. Plotting first example (index 0).")

    # Generate the prediction plot for the selected example
    # The plot_prediction function handles the quantiles automatically
    fig, ax = plt.subplots(figsize=(12, 7))
    model.plot_prediction(
        raw_predictions.x, 
        raw_predictions.output, 
        idx=best_example_idx, 
        add_loss_to_title=True,
        ax=ax
    )
    ndc_id = val_loader.dataset.data.loc[best_example_idx, "ndc11"].values[0]
    ax.set_title(f"TFT Forecast vs Actuals for NDC: {ndc_id}")
    fig.tight_layout()

    # 4. Save the Forecasting Plot
    fig.savefig(FORECAST_PLOT_PATH, dpi=300)
    logging.info(f"✅ Saved forecast plot to '{FORECAST_PLOT_PATH}'")
    plt.close(fig)

    logging.info("✅ Explanation process complete.")


if __name__ == '__main__':
    main()
