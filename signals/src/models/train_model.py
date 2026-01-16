import polars as pl
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
PROCESSED_PATH = "data/processed"
MODELS_PATH = "src/models/artifacts"
os.makedirs(MODELS_PATH, exist_ok=True)

LOOKAHEAD_WEEKS = 8
SPIKE_THRESHOLD = 0.10


def train_advanced_model():
    print("🚀 Starting Advanced Model Training...")

    # 1. Load the "Kitchen Sink" Features
    try:
        df = pl.read_parquet(os.path.join(
            PROCESSED_PATH, "weekly_features.parquet"))
        print(f"   📂 Loaded Data: {df.height:,} rows")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # 2. Create Target (Future Price Spike)
    print("   🔮 creating Targets...")
    df = df.sort(["ndc11", "effective_date"])

    df = df.with_columns([
        pl.col(
            "price_per_unit").shift(-LOOKAHEAD_WEEKS).over("ndc11").alias("future_price")
    ]).filter(pl.col("future_price").is_not_null())

    df = df.with_columns([
        ((pl.col("future_price") - pl.col("price_per_unit")) /
         pl.col("price_per_unit")).alias("pct_change")
    ])

    df = df.with_columns([
        (pl.col("pct_change") > SPIKE_THRESHOLD).cast(pl.Int8).alias("target")
    ])

    # 3. Train/Test Split (Time-Based)
    print("   ✂️  Splitting Data...")
    split_date = pl.date(2024, 1, 1)
    train = df.filter(pl.col("effective_date") < split_date)
    test = df.filter(pl.col("effective_date") >= split_date)

    # 4. Define Features
    features = [
        "is_shortage",
        "weeks_in_shortage",
        "price_velocity_4w",    # NEW
        "price_volatility_12w",  # NEW
        "market_hhi",           # NEW
        "num_competitors"       # NEW
    ]

    X_train = train.select(features).to_pandas()
    y_train = train.select("target").to_pandas()
    X_test = test.select(features).to_pandas()
    y_test = test.select("target").to_pandas()

    # 5. Train XGBoost
    print("   🧠 Training Advanced XGBoost...")

    # Calculate scale_pos_weight to handle imbalance
    pos_count = y_train["target"].sum()
    neg_count = len(y_train) - pos_count
    weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=weight,
        n_estimators=200,      # More trees
        learning_rate=0.05,    # Slower learning for better generalization
        # Deeper trees to capture interactions (Monopoly + Shortage)
        max_depth=6,
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )

    model.fit(X_train, y_train)

    # 6. Evaluate
    print("\n   📝 EVALUATION REPORT:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    print("\n   📊 FEATURE IMPORTANCE:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        print(f"      {features[i]}: {importances[i]:.4f}")

    # 7. Save
    with open(os.path.join(MODELS_PATH, "spike_predictor_v2.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("   💾 Advanced Model Saved!")


if __name__ == "__main__":
    train_advanced_model()
