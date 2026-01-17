import sys
import os
import polars as pl
import numpy as np

# 1. Setup path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.simulation.monte_carlo import RiskSimulator
    print("✅ Found 'RiskSimulator' class.")
except ImportError:
    print("❌ Error: Could not find 'RiskSimulator' in src/simulation/monte_carlo.py")
    sys.exit(1)


def test_simulation_logic():
    print("\n🎲 Starting Monte Carlo Logic Test...")

    # 2. Mock Data (Polars DataFrame)
    # Scenario:
    # - Drug A: $100k spend, Risk 10/10 (Guaranteed Spike)
    # - Drug B: $100k spend, Risk 0/10 (Guaranteed Safe)
    # - Drug C: $100k spend, Risk 5/10 (Coin Flip)

    mock_portfolio = pl.DataFrame({
        "drug_name": ["RiskyMeds", "SafeMed", "CoinFlipMed"],
        "current_spend": [100_000.0, 100_000.0, 100_000.0],
        "manufacturer_risk_score": [10, 0, 5]
    })

    print(f"   📊 Input Portfolio:\n{mock_portfolio}")

    # 3. Initialize Simulator
    simulator = RiskSimulator(num_simulations=1000)

    print("   ⚙️ Running 1,000 Simulations...")
    results = simulator.simulate(mock_portfolio)

    # 4. Analyze Results
    print("\n   🔎 Simulation Results:")
    print(results)

    # 5. Assertions (The Proof)

    # CHECK 1: SafeMed should have $0 loss
    safe_loss = results.filter(pl.col("drug_name") == "SafeMed")[
        "mean_loss"][0]
    if safe_loss == 0:
        print("   ✅ PASS: SafeMed has $0 projected loss.")
    else:
        print(f"   ❌ FAIL: SafeMed has loss ({safe_loss}) but risk was 0!")

    # CHECK 2: RiskyMeds should be ~ $50,000 loss (50% spike on $100k)
    # Since Probability is 100%, every run triggers a 50% hike.
    risky_loss = results.filter(pl.col("drug_name") == "RiskyMeds")[
        "mean_loss"][0]
    expected_loss = 50_000.0  # 50% of 100k

    # Allow small float error
    if abs(risky_loss - expected_loss) < 1.0:
        print(
            f"   ✅ PASS: RiskyMeds loss is exactly ${risky_loss:,.2f} (Implies 100% strike rate).")
    else:
        print(
            f"   ⚠️ CHECK: RiskyMeds loss is ${risky_loss:,.2f} (Expected ~$50k).")

    # CHECK 3: CoinFlipMed should be roughly $25,000 loss (50% chance of 50k loss)
    # Monte Carlo variance means it won't be exact, but should be close.
    coin_loss = results.filter(pl.col("drug_name") == "CoinFlipMed")[
        "mean_loss"][0]
    if 20_000 < coin_loss < 30_000:
        print(
            f"   ✅ PASS: CoinFlipMed loss is ${coin_loss:,.2f} (Within statistical variance of $25k).")
    else:
        print(f"   ⚠️ WARNING: CoinFlipMed variance high (${coin_loss:,.2f}).")


if __name__ == "__main__":
    test_simulation_logic()
