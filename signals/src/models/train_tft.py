import os
import sys
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
import pandas as pd
import logging

# --- FIX: Register Safe Globals for PyTorch 2.6+ ---
# We must allowlist both PyTorch Forecasting custom classes AND Pandas objects
# because the checkpoints contain the dataset configuration which uses them.
torch.serialization.add_safe_globals([
    GroupNormalizer,
    NaNLabelEncoder,
    pd.DataFrame,
    pd.Series,
    pd.Index,
    pd.RangeIndex,
    pd.core.indexes.base.Index,
    pd.core.indexes.range.RangeIndex
])

# --- Fix Path for Imports ---
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# --- Local Imports ---
try:
    from src.models.tft_loader import create_tft_dataloaders
except ImportError:
    try:
        from tft_loader import create_tft_dataloaders
    except ImportError:
        logging.error(
            "Critical Error: Could not import data loader. Check your python path.")
        exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
pl.seed_everything(42)

# Model Artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "src/models/artifacts")
MODEL_CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "tft_model.ckpt")

# Training Hyperparameters
MAX_EPOCHS = 30
GRADIENT_CLIP_VAL = 0.1
DEFAULT_LEARNING_RATE = 0.03
FIND_LEARNING_RATE = True

# Model Architecture
HIDDEN_SIZE = 16
ATTENTION_HEADS = 4
DROPOUT = 0.1


def main():
    logging.info("🚀 Starting Temporal Fusion Transformer training process...")

    # 1. Create DataLoaders
    try:
        train_loader, val_loader, training_dataset = create_tft_dataloaders(
            batch_size=128)
    except Exception as e:
        logging.error(f"❌ Data Error: {e}")
        return

    # 2. Configure Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=False,
        mode="min"
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ARTIFACTS_DIR,
        filename="tft_model",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    logging.info(f"Model checkpoints will be saved to '{ARTIFACTS_DIR}'.")

    # 3. Initialize Model
    logging.info("Initializing TFT model...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_SIZE,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        optimizer="Adam",
    )

    # 4. Configure Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # 5. Find Learning Rate
    if FIND_LEARNING_RATE:
        logging.info("Finding optimal learning rate...")
        try:
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                tft,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                max_lr=0.1,
                min_lr=1e-5,
            )

            suggested_lr = lr_finder.suggestion()

            if suggested_lr is None:
                logging.warning(
                    "⚠️ LR Finder returned None. Using default LR.")
                tft.hparams.learning_rate = DEFAULT_LEARNING_RATE
            else:
                logging.info(f"📈 Suggested learning rate: {suggested_lr:.2e}")
                tft.hparams.learning_rate = suggested_lr

        except Exception as e:
            logging.warning(
                f"⚠️ Learning rate finder failed: {e}. Using default LR.")
            tft.hparams.learning_rate = DEFAULT_LEARNING_RATE
    else:
        tft.hparams.learning_rate = DEFAULT_LEARNING_RATE

    # 6. Train
    logging.info("Fitting the model...")
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    logging.info(
        f"\n✅ Training complete. Best model saved to: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
