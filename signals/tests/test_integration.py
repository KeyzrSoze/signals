import sys
import os
import polars as pl
from datetime import date

# --- Path Correction ---
# Add the project's root directory (the one containing the 'signals' package) to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use an absolute import from the project root
try:
    from signals.src.features.signal_generator import integrate_sentinel_risk
    print("✅ Found 'integrate_sentinel_risk' function.")
except ImportError:
    print("❌ Error: Could not find 'integrate_sentinel_risk' in signals/src/features/signal_generator.py")
    sys.exit(1)


def test_time_machine():
    print("\n🧪 Starting Time Machine Integration Test...")

    # 3. Create a Dummy Feature Table (Simulating a Price Row)
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
        assert False, f"Function execution failed: {e}"

    # 5. Inspect the Results
    print("\n   🔎 Resulting Data:")
    print(enriched_df)

    # 6. Verification Logic
    assert "manufacturer_risk_score" in enriched_df.columns, "Output dataframe is missing 'manufacturer_risk_score'"
    
    abbott_row = enriched_df.filter(pl.col("manufacturer") == "Abbott")
    safe_row = enriched_df.filter(pl.col("manufacturer") == "SafeMeds Inc")

    assert abbott_row.height > 0, "Abbott row was lost during the join."
    
    score = abbott_row["manufacturer_risk_score"][0]
    print(f"\n   🎯 Abbott Risk Score: {score}")

    assert score > 0, "Abbott's risk score should be > 0, as a risk event should have been found."
    print("   ✅ SUCCESS: The model successfully looked back in time and found the risk for Abbott!")

    # Verify SafeMeds
    safe_score = safe_row["manufacturer_risk_score"][0] if safe_row.height > 0 else -1
    assert safe_score == 0, f"SafeMeds Inc should have a risk score of 0, but got {safe_score}"
    print("   ✅ SUCCESS: Safe manufacturer correctly has 0 risk.")


if __name__ == "__main__":
    # This assumes a `sentinel_risks.parquet` file exists in `data/processed`
    # with a historical risk for 'Abbott'
    test_time_machine()
