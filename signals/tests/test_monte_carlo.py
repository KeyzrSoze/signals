import sys
import os
import polars as pl
import numpy as np

# --- Path Correction ---
# Add the project's root directory (the one containing the 'signals' package) to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use an absolute import from the project root
try:
    from signals.src.simulation.monte_carlo import RiskSimulator
    print("✅ Found 'RiskSimulator' class.")
except ImportError:
    print("❌ Error: Could not find 'RiskSimulator' in signals/src/simulation/monte_carlo.py")
    sys.exit(1)


def test_updated_simulation_logic():
    print("\n🎲 Starting Monte Carlo Inflation Forecast Logic Test...")

    # 2. Mock Data with new schema for `forecast_inflation`
    # Scenarios:
    # - Guaranteed Hit: High risk score, P90 forecast > current price.
    # - Guaranteed Miss: High risk score, but P90 forecast < current price (no loss).
    # - No Risk: Zero risk score, so spike should never happen.
    # - Coin Flip: 50/50 risk score with a positive P90 forecast.
    mock_portfolio = pl.DataFrame({
        "drug_name": ["GuaranteedHit", "GuaranteedMiss", "NoRisk", "CoinFlip"],
        "current_spend": [100_000.0, 100_000.0, 100_000.0, 100_000.0],
        "current_price": [10.0, 10.0, 10.0, 10.0],
        "manufacturer_risk_score": [10, 10, 0, 5],
        "forecast_p90": [15.0, 8.0, 15.0, 15.0],
    })

    print(f"   📊 Input Portfolio:\n{mock_portfolio}")

    # 3. Initialize Simulator
    simulator = RiskSimulator(num_simulations=10000) # Use more sims for stable stats

    print("   ⚙️ Running 10,000 Simulations...")
    results = simulator.forecast_inflation(mock_portfolio)

    # 4. Analyze Results
    print("\n   🔎 Simulation Results:")
    print(results)

    # 5. Assertions
    # CHECK 1: GuaranteedHit should have a mean loss based on the P90 forecast.
    # Units = 100k/10 = 10k. Unit Loss = 15-10=5. Shock Loss = 50k. Prob = 100%.
    hit_loss = results.filter(pl.col("drug_name") == "GuaranteedHit")["mean_loss"][0]
    expected_hit_loss = 50_000.0
    assert abs(hit_loss - expected_hit_loss) < 1.0, f"GuaranteedHit loss should be exactly {expected_hit_loss}, but was {hit_loss}"
    print(f"   ✅ PASS: GuaranteedHit loss is ${hit_loss:,.2f}.")

    # CHECK 2: GuaranteedMiss should have zero loss, as P90 is below current price.
    miss_loss = results.filter(pl.col("drug_name") == "GuaranteedMiss")["mean_loss"][0]
    assert miss_loss == 0.0, f"GuaranteedMiss loss should be 0, but was {miss_loss}"
    print("   ✅ PASS: GuaranteedMiss has $0 projected loss as P90 < current price.")

    # CHECK 3: NoRisk should have zero loss, as probability is 0.
    norisk_loss = results.filter(pl.col("drug_name") == "NoRisk")["mean_loss"][0]
    assert norisk_loss == 0.0, f"NoRisk loss should be 0, but was {norisk_loss}"
    print("   ✅ PASS: NoRisk has $0 projected loss as risk score was 0.")

    # CHECK 4: CoinFlip should have a mean loss of roughly 50% of the shock loss.
    # Shock Loss = 50k. Prob = 50%. Expected mean loss ~ 25k.
    coin_loss = results.filter(pl.col("drug_name") == "CoinFlip")["mean_loss"][0]
    expected_coin_loss = 25_000.0
    # Allow for statistical variance
    assert 22_000 < coin_loss < 28_000, f"CoinFlip loss is {coin_loss}, outside expected statistical range of ~$25k"
    print(f"   ✅ PASS: CoinFlip loss is ${coin_loss:,.2f} (Within statistical variance of $25k).")


if __name__ == "__main__":
    test_updated_simulation_logic()
