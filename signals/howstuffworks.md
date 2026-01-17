# Signals: System Architecture & Logic

## 1. The "Elevator Pitch"
**What does this system do?**
"Signals" is a "Digital Twin" of the Pharmaceutical Supply Chain. It maps the physical web of factories, parent companies, and shipping lanes to predict failures before they happen.

**How does it work?**
1.  **It Listens (24/7):** An autonomous "Reflex Agent" monitors FDA feeds every hour.
2.  **It Thinks (Weekly):** A Deep Learning "Brain" wakes up weekly to retrain on global pricing trends.
3.  **It Predicts:** It combines real-time alerts with long-term forecasts to protect the client's budget.

---

## 2. The Engineering Deep Dive

### A. The Knowledge Graph (The "Map")
* **Technology:** Neo4j + Gemini 2.5 Flash.
* **Logic:** Maps `(Factory)-[:MAKES]->(Ingredient)-[:IN]->(Drug)`. Uses **Risk Propagation** to trace "Contagion" from a failed factory to specific drug products.

### B. The Predictive Core (Temporal Fusion Transformer)
* **Architecture:** A **Temporal Fusion Transformer (TFT)** that replaces standard XGBoost.
* **Capability:** Provides **Probabilistic Forecasting** (P10/P90 confidence intervals) and understands seasonality (e.g., Flu Season).

### C. The Shadow Formulary (Financial Simulation)
* **Objective:** Calculates financial **Value at Risk (VaR)**.
* **Logic:** `Loss = (P90_Forecast - Current_Price) * Volume`. Allows CFOs to budget for "Inflation Shock."

### D. The Sentinel (Unstructured Intelligence)
* **Role:** A "Signal Transducer" using Gemini Flash to convert messy FDA text into quantitative `severity_scores` (0-10).

### E. The Autonomous Nervous System (New!)
**Architecture:**
We split the system into two biological components using **Celery** (Scheduler) and **Redis** (Memory).

1.  **The Reflex System (The Watchdog):**
    * **Frequency:** Every 60 Minutes.
    * **Task:** Checks FDA RSS feeds for *new* Warning Letters only.
    * **Logic:** If Gemini scores an event > 8/10 (e.g., "Factory Shutdown"), it triggers an **Immediate Alert** without waiting for the weekly report.
    * **Cost:** Near zero (sleeps 59 mins/hour).

2.  **The Brain System (The Strategist):**
    * **Frequency:** Every Wednesday (06:00 UTC).
    * **Task:** Runs the heavy pipeline: `Ingest Prices` -> `Update Graph` -> `Retrain TFT Model` -> `Update Financial Forecasts`.
    * **Purpose:** Captures slow-moving trends like inflation or market consolidation.