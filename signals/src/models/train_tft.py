import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import logging

# --- Local Imports ---
# Assumes this script is run from the project root, or the src folder is in the python path
try:
    from tft_loader import create_tft_dataloaders
except ImportError:
    logging.error("Failed to import 'create_tft_dataloaders'. Make sure you are running from the project root.")
    logging.error("Attempting relative import for execution within src/models...")
    from .tft_loader import create_tft_dataloaders


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pl.seed_everything(42)  # for reproducibility

# Model Artifacts
ARTIFACTS_DIR = "signals/src/models/artifacts"
MODEL_CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "tft_model.ckpt")

# Training Hyperparameters
MAX_EPOCHS = 30
GRADIENT_CLIP_VAL = 0.1
DEFAULT_LEARNING_RATE = 0.03
FIND_LEARNING_RATE = True  # Set to False to skip LR finder and use the default

# Model Architecture
HIDDEN_SIZE = 16
ATTENTION_HEADS = 4
DROPOUT = 0.1


def main():
    """
    Main function to orchestrate the training of the Temporal Fusion Transformer.
    """
    logging.info("🚀 Starting Temporal Fusion Transformer training process...")

    # 1. Create DataLoaders from our custom loader script
    try:
        train_loader, val_loader, training_dataset = create_tft_dataloaders(batch_size=128)
    except FileNotFoundError as e:
        logging.error(f"❌ {e}")
        logging.error("   Cannot train model without data. Please run feature generation scripts first.")
        return

    # 2. Configure PyTorch Lightning Callbacks
    # Stop training early if validation loss does not improve
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    
    # Save the best model checkpoint based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=ARTIFACTS_DIR,
        filename="tft_model",  # Saved as tft_model.ckpt
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    logging.info(f"Model checkpoints will be saved to '{ARTIFACTS_DIR}'.")

    # 3. Initialize the Temporal Fusion Transformer Model
    logging.info("Initializing TFT model...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_SIZE, # From documentation, often same as hidden_size
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]), # Predict P10, P50, P90
        optimizer="Adam",
        # learning_rate is configured below by the LR finder
    )
    logging.info(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # 4. Configure the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Use GPU if available, else CPU
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # 5. Find Optimal Learning Rate (optional but recommended)
    if FIND_LEARNING_RATE:
        logging.info("Finding optimal learning rate...")
        try:
            lr_finder = trainer.tuner.lr_find(
                tft,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                max_lr=0.1,
                min_lr=1e-5,
            )
            suggested_lr = lr_finder.suggestion()
            logging.info(f"📈 Suggested learning rate: {suggested_lr:.2e}")
            tft.hparams.learning_rate = suggested_lr
        except Exception as e:
            logging.warning(f"⚠️ Learning rate finder failed: {e}. Using default LR: {DEFAULT_LEARNING_RATE}")
            tft.hparams.learning_rate = DEFAULT_LEARNING_RATE
    else:
        tft.hparams.learning_rate = DEFAULT_LEARNING_RATE

    # 6. Start the Training Loop
    logging.info("Fitting the model...")
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    logging.info(f"\n✅ Training complete. Best model saved to: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    # Ensure the directory for model artifacts exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    main()
