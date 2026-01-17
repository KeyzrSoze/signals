import sys
import os
import polars as pl
from datetime import date

# 1. Setup path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# 2. Try to import the new function
try:
    from src.features.signal_generator import integrate_sentinel_risk
    print("✅ Found 'integrate_sentinel_risk' function.")
except ImportError:
    print("❌ Error: Could not find 'integrate_sentinel_risk' in src/features/signal_generator.py")
    sys.exit(1)


def test_time_machine():
    print("\n🧪 Starting Time Machine Integration Test...")

    # 3. Create a Dummy Feature Table (Simulating a Price Row)
    # CHANGE: Renamed 'date' to 'effective_date' to match the project schema
    print("   📊 Creating dummy price data for 'Abbott' (Post-Recall)...")

    mock_features = pl.DataFrame({
        "effective_date": [date(2026, 1, 20), date(2026, 1, 20)],
        "manufacturer": ["Abbott", "SafeMeds Inc"],
        "price": [100.0, 50.0]
    })

    # Polars join_asof requires sorted keys
    mock_features = mock_features.sort("effective_date")

    # 4. Run the Integration
    print("   ⚙️ Running 'integrate_sentinel_risk'...")
    try:
        enriched_df = integrate_sentinel_risk(mock_features)
    except Exception as e:
        print(f"   ❌ Execution Failed: {e}")
        # Debug helper: print columns if it fails
        print(f"   (Input Columns: {mock_features.columns})")
        return

    # 5. Inspect the Results
    print("\n   🔎 Resulting Data:")
    print(enriched_df)

    # 6. Verification Logic
    # We expect Abbott to have a score (likely 9, based on your previous run)
    abbott_row = enriched_df.filter(pl.col("manufacturer") == "Abbott")
    safe_row = enriched_df.filter(pl.col("manufacturer") == "SafeMeds Inc")

    if abbott_row.height > 0:
        # Handle cases where column might be 'manufacturer_risk_score' or 'risk_score'
        # The AI assistant likely named it 'manufacturer_risk_score' based on the prompt
        target_col = "manufacturer_risk_score" if "manufacturer_risk_score" in enriched_df.columns else "risk_score"

        score = abbott_row[target_col][0]
        print(f"\n   🎯 Abbott Risk Score: {score}")

        if score > 0:
            print(
                "   ✅ SUCCESS: The model successfully looked back in time and found the risk!")
        else:
            print(
                "   ⚠️ WARNING: Abbott score is 0. Check if 'sentinel_risks.parquet' has data for Abbott.")
            print(
                "   (Note: Ensure your parquet file has 'Abbott' exactly, case-sensitive)")
    else:
        print("   ❌ FAIL: Abbott row vanished?")

    # Verify SafeMeds
    target_col = "manufacturer_risk_score" if "manufacturer_risk_score" in enriched_df.columns else "risk_score"
    safe_score = safe_row[target_col][0] if safe_row.height > 0 else 0
    if safe_score == 0:
        print("   ✅ SUCCESS: Safe manufacturer correctly has 0 risk.")


if __name__ == "__main__":
    test_time_machine()
