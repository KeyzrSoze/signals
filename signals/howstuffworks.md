# Signals: System Architecture & Logic

## 1. The "Elevator Pitch" (Layman Explanation)
**What does this system do?**
"Signals" is an Early Warning System for pharmacy procurement. Just as a weather radar predicts a storm before it hits, Signals predicts drug price spikes before they hit the invoice.

**How does it work?**
Most systems look at *history* (what happened last month). Signals looks at *pressure* (what is happening right now).
1.  **It Listens:** It monitors government databases for factory inspections, recalls, and competitor exits 24/7.
2.  **It Reads:** An AI (Gemini) reads thousands of messy FDA reports to find "hidden" risks—like a factory shutdown in India—that human analysts miss.
3.  **It Predicts:** It combines these risk signals with market data to calculate a probability score (0-100%) that a specific drug will double in price in the next 8 weeks.

**The Result:**
Instead of reacting to a price hike, we proactively switch to a cheaper alternative 3 weeks early, saving the client millions in "inflation avoidance."

---

## 2. The Engineering Deep Dive (Technical Explanation)
**Architecture Pattern:**
The system utilizes a **Lambda Architecture** variant, combining batch processing for historical baselines with an event-driven "Speed Layer" for real-time risk ingestion.

**The 3 Core Components:**

### A. The Ingestion Engine (ETL)
* **Polars-based Pipelines:** We use Rust-backed Polars DataFrames for high-performance ETL. Unlike Pandas, this allows us to process 5+ years of NADAC (CMS) pricing data in milliseconds using lazy evaluation.
* **Event Sourcing:** FDA shortage data is treated as an immutable stream of state changes (Start -> Update -> Resolved), allowing us to reconstruct the "State of the World" at any point in history without data leakage.

### B. The Sentinel (Unstructured Intelligence)
* **LLM Integration:** We utilize **Google Gemini 2.5 (Flash)** as a "Reasoning Engine."
* **Signal Transduction:** The system ingests unstructured text (RSS feeds, Warning Letters). The LLM acts as a transducer, converting qualitative text into quantitative `severity_scores` (0-10) and extracting entities (Manufacturers) via strict JSON schema enforcement.
* **Circuit Breakers:** The pipeline includes robust error handling and fallback mechanisms to ensure downstream models never fail due to API outages.

### C. The Predictive Core (XGBoost)
* **Point-in-Time Correctness:** We utilize `join_asof` logic to align features. This ensures that when the model trains on a date (e.g., Jan 1, 2024), it *only* sees data available up to Dec 31, 2023. This eliminates "Look-ahead Bias."
* **Hybrid Feature Store:** The model consumes a "Kitchen Sink" of features:
    * **Market Structure:** Herfindahl-Hirschman Index (HHI) to measure monopoly power.
    * **Momentum:** Price velocity and acceleration derivatives.
    * **External Shock:** The aggregated `risk_score` from the Sentinel.

**The Stack:**
* **Language:** Python 3.12
* **Compute:** Polars (Local), Scalable to Spark/BigQuery.
* **AI:** Google GenAI SDK (Gemini 2.5).
* **ML:** XGBoost (Gradient Boosted Trees).

### D. The Shadow Formulary (Financial Simulation)
* **Objective:** Translate abstract probabilities into concrete financial exposure (Value at Risk).
* **Stochastic Modeling:** We utilize a **Monte Carlo Engine** that runs 1,000 parallel market simulations per drug.
* **Logic:**
    * The engine takes the `risk_score` (e.g., 9/10) and treats it as a probability threshold.
    * It "rolls the dice" 1,000 times to see how many times the supply chain breaks.
* **Output:** It calculates the **Expected Loss** (Mean) and the **Worst Case Scenario** (95th Percentile VaR). This allows CFOs to see: *"We have a 90% chance of losing $2M on this drug next quarter."*