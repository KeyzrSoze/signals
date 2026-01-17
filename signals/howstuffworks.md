# Signals: System Architecture & Logic

## 1. The "Elevator Pitch"
**What does this system do?**
"Signals" is a "Digital Twin" of the Pharmaceutical Supply Chain. It doesn't just track prices; it maps the physical web of factories, parent companies, and shipping lanes to predict failures before they happen.

**The "Crystal Ball" Logic:**
1.  **The Ear:** We listen to FDA signals (Recalls, Inspections).
2.  **The Brain:** A Knowledge Graph (Neo4j) traces that signal from a factory in India to every specific pill bottle on a US pharmacy shelf.
3.  **The Prediction:** AI models combine this "Structural Risk" with "Price Momentum" to forecast spikes 8 weeks out.

---

## 2. The Engineering Deep Dive

### A. The Knowledge Graph (The "Map")
* **Technology:** Neo4j + Gemini 2.5 Flash.
* **The Problem:** Pharma data is opaque. "Teva USA" and "Teva Israel" look like different companies in a database, but they share risk.
* **The Solution:** We built a Graph Topology:
    * `(Factory)-[:MAKES]->(Ingredient)-[:IN]->(Drug)`
    * `(Parent Corp)-[:OWNS]->(Subsidiary)`
* **Risk Propagation:** If a Factory node turns "Red" (Inspection Failure), our **Graph Traversal Algorithm** pushes that risk score down the edges. If you buy a drug connected to that factory, your risk score goes up, even if the price hasn't moved yet.

### B. The Ingestion Engine (ETL)
* **Polars-based Pipelines:** High-performance Rust backends for processing millions of NADAC rows.
* **Data Sources:**
    * **CMS NADAC:** The "Ticker Tape" of pharmacy pricing.
    * **FDA Shortages:** The historical record of failures.
    * **FDA FEI (Facility Establishment Identifier):** The physical address book of every drug factory on earth.

### C. The Predictive Core (Hybrid AI)
* **Feature Fusion:** We combine two types of intelligence:
    1.  **Temporal:** Price Velocity, Volatility (Time Series).
    2.  **Structural:** Graph Centrality, Supplier Concentration (Network Science).
* **Model:** XGBoost Classifier (Gradient Boosted Trees).
* **Zero Leakage:** Strict `join_asof` usage ensures we never train on data from the future.

### D. The Sentinel (LLM Agent)
* **Role:** The Reader.
* **Logic:** FDA reports are messy text ("...debris found in hopper 4...").
* **Agent:** We use **Gemini 2.5 Flash** to read these reports and extract:
    * **Entity:** Who is it? (Mapped to Graph Node).
    * **Severity:** 0-10 Score.
    * **Type:** "Quality Failure" vs "Paperwork Error".

### E. Financial Simulation
* **Monte Carlo:** We translate "Risk Score 8" into "Expected Loss: $450,000".
* **Method:** 1,000 parallel simulations per drug per week.