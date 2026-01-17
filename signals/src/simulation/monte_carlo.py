import numpy as np
import polars as pl
from typing import Dict, List


class RiskSimulator:
    """
    Runs a Monte Carlo simulation to forecast potential inflation impact on a drug portfolio,
    using both a risk score for event probability and a TFT model's forecast for event magnitude.
    """
    def __init__(self, num_simulations: int = 1000, spike_probability_scale: float = 0.1):
        """
        Initialize the Simulator.

        Args:
            num_simulations: How many parallel universes to simulate (default 1000).
            spike_probability_scale: Scaling factor for the risk score. 
                                     e.g., if risk score is 0-10, scale=0.1 means 
                                     Score 5 -> 0.5 (50%) probability.
        """
        self.num_simulations = num_simulations
        self.scale = spike_probability_scale

    def forecast_inflation(self, portfolio_df: pl.DataFrame) -> pl.DataFrame:
        """
        Run the Monte Carlo simulation on a portfolio of drugs to forecast inflation.

        Args:
            portfolio_df: A Polars DataFrame with columns:
                          ['drug_name', 'current_spend', 'current_price', 
                           'manufacturer_risk_score', 'forecast_p90']

        Returns:
            A Polars DataFrame with per-drug simulation results, including mean
            and worst-case (95th percentile) financial loss.
        """
        # Validate input columns
        required_cols = ['drug_name', 'current_spend', 'current_price', 'manufacturer_risk_score', 'forecast_p90']
        if not all(col in portfolio_df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        results = []
        drugs = portfolio_df.to_dicts()

        for drug in drugs:
            # 1. Calculate Probability of an inflation event happening
            prob_spike = min(drug['manufacturer_risk_score'] * self.scale, 1.0)

            # 2. Calculate the new loss magnitude based on the TFT forecast
            # Avoid division by zero if current_price is not positive
            if drug['current_price'] > 0:
                units_purchased = drug['current_spend'] / drug['current_price']
            else:
                units_purchased = 0
            
            projected_unit_loss = max(0, drug['forecast_p90'] - drug['current_price'])
            total_loss_on_shock = projected_unit_loss * units_purchased

            # 3. Run Simulations (Vectorized with NumPy for speed)
            rng = np.random.random(self.num_simulations)
            has_spike = rng < prob_spike
            simulated_losses = np.where(has_spike, total_loss_on_shock, 0.0)

            # 4. Aggregate Statistics
            mean_loss = np.mean(simulated_losses)
            worst_case_loss = np.percentile(simulated_losses, 95) # 95th Percentile VaR

            results.append({
                "drug_name": drug['drug_name'],
                "current_spend": drug['current_spend'],
                "manufacturer_risk_score": drug['manufacturer_risk_score'],
                "forecast_p90": drug['forecast_p90'],
                "mean_loss": mean_loss,
                "worst_case_loss": worst_case_loss,
                "prob_spike": prob_spike
            })

        return pl.DataFrame(results)

if __name__ == '__main__':
    # --- Example usage of the updated engine ---
    
    # Create a sample DataFrame with the new required columns
    sample_data = {
        'drug_name': ['Drug A', 'Drug B', 'Drug C'],
        'current_spend': [1_000_000, 500_000, 200_000],
        'current_price': [100.0, 50.0, 40.0],
        'manufacturer_risk_score': [8, 3, 5], # 80%, 30%, and 50% chance of spike
        'forecast_p90': [160.0, 55.0, 38.0] # AI predicts worst-case prices
    }
    portfolio = pl.DataFrame(sample_data)

    print("--- Input Portfolio ---")
    print(portfolio)

    # Initialize and run the simulator
    simulator = RiskSimulator(num_simulations=10000) # More sims for smoother results
    inflation_forecast = simulator.forecast_inflation(portfolio)
    
    print("\n--- Inflation Forecast Results ---")
    print(inflation_forecast)

    # --- Example Interpretation ---
    # For Drug A:
    # Units = 1,000,000 / 100 = 10,000
    # Unit Loss = max(0, 160 - 100) = $60
    # Total Loss on Shock = 10,000 * $60 = $600,000
    # Mean Loss should be approx. 80% (prob_spike) * $600,000 = $480,000
    
    # For Drug C:
    # Unit Loss = max(0, 38 - 40) = $0. The P90 forecast is below the current price.
    # Therefore, all loss metrics should be 0.
