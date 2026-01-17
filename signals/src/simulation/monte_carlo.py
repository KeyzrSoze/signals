import numpy as np
import polars as pl
from typing import Dict, List


class RiskSimulator:
    def __init__(self, num_simulations: int = 1000, spike_probability_scale: float = 0.1):
        """
        Initialize the Simulator.

        Args:
            num_simulations: How many parallel universes to simulate (default 1000).
            spike_probability_scale: Scaling factor. 
                                     If risk score is 0-10, scale=0.1 means 
                                     Score 5 -> 0.5 (50%) probability.
        """
        self.num_simulations = num_simulations
        self.scale = spike_probability_scale

    def simulate(self, portfolio_df: pl.DataFrame) -> pl.DataFrame:
        """
        Run the Monte Carlo simulation on a portfolio of drugs.
        """
        results = []

        # Convert to dictionary for faster iteration than pure Polars looping
        drugs = portfolio_df.to_dicts()

        for drug in drugs:
            name = drug['drug_name']
            spend = drug['current_spend']
            risk_score = drug['manufacturer_risk_score']

            # 1. Calculate Probability of Inflation Event
            # Logic: Score 9 * 0.1 = 0.9 (90% chance)
            prob_spike = min(risk_score * self.scale, 1.0)

            # 2. Run Simulations (Vectorized with NumPy for speed)
            # Create an array of random numbers [0.0 to 1.0]
            rng = np.random.random(self.num_simulations)

            # If random number < probability, the spike happens
            # Spike magnitude is fixed at 50% (1.5x multiplier) for this MVP
            has_spike = rng < prob_spike

            # Calculate financial impact
            # If spike: loss = spend * 0.5. If no spike: loss = 0.
            simulated_losses = np.where(has_spike, spend * 0.5, 0.0)

            # 3. Aggregate Stats
            mean_loss = np.mean(simulated_losses)
            # 95th Percentile VaR (Value at Risk)
            worst_case_loss = np.percentile(simulated_losses, 95)

            results.append({
                "drug_name": name,
                "current_spend": spend,
                "manufacturer_risk_score": risk_score,
                "mean_loss": mean_loss,
                "worst_case_loss": worst_case_loss,
                "prob_spike": prob_spike
            })

        return pl.DataFrame(results)
