# Project Signals: Master Canvas

**Project Goal:** Construct an enterprise-grade Early Warning System (EWS) for pharmacy price spikes using **Graph Neural Networks** & **Probabilistic Deep Learning**.
**Core Metric:** Accuracy (Zero Data Leakage).
**Tech Stack:** * **Core:** Python 3.12, Polars, Git.
* **Graph:** Neo4j (Graph Database), Cypher.
* **AI/ML:** PyTorch Forecasting (Temporal Fusion Transformer), XGBoost.
* **GenAI:** Google Gemini 2.5 Flash (Entity Resolution & Sentiment).

## Phase 1: Environment & Architecture
- [x] Structure: Domain-driven directory tree.
- [x] Environment: Conda environment `signals` active.
- [x] Version Control: Git initialized.

## Phase 2: The Ingestion Engine
- [x] **2A: NADAC Processor:** Strict schema enforcement & case-insensitive normalization.
- [x] **2B: FDA Shortage Scraper:** Longitudinal "Event Stream" (Start/Update/End).
- [x] **2C: Reference Data:** Master list of valid NDCs.

## Phase 3: The Entity Map (Graph Upgrade)
- [x] **Entity Map Construction:** Created bridge table (NDC <-> Ingredient <-> Manufacturer).
- [x] **Graph Database Hydration:** Ported flat maps into **Neo4j** nodes/edges.
- [x] **Facility Enrichment:** Used **Gemini Flash** to link corporate subsidiaries to physical FDA-registered factories (FEI Numbers).

## Phase 4: Signal Generation
- [x] **Time Machine Logic:** `join_asof` for point-in-time correctness.
- [x] **Graph Topology:** Extracted **FastRP Embeddings** and Supplier Diversity Scores to measure supply chain resilience.
- [x] **Sentinel Integration:** Merged unstructured LLM risk scores with structured market data.

## Phase 5: Baseline Modeling (XGBoost)
- [x] **Binary Classification:** Trained XGBoost to predict >10% spikes.
- [x] **Performance:** Achieved ~68% Recall on historical baselines.

## Phase 6: Consulting Deliverables
- [x] **Automated Risk Reporting:** "Monday Morning" watchlist ranking drugs by risk probability.

## Phase 7: The Sentinel (Unstructured Intelligence)
- [x] **LLM "Signal Transducer":** Gemini Flash parses FDA text into quantitative `severity_scores` (0-10).

## Phase 8: Financial Risk Simulation (Shadow Formulary)
- [x] **Inflation Forecasting:** Upgraded from static risk scoring to dynamic **Inflation Forecasting**.
- [x] **Logic:** Uses TFT P90 (Worst Case) forecasts to calculate `projected_unit_loss`.

## Phase 9: System Integration
- [x] **Temporal Joining:** Unified Graph features, Price history, and Sentiment scores into a single `TimeSeriesDataSet`.

## Phase 10: Deep Learning (Blueprint 3)
- [x] **Temporal Fusion Transformer (TFT):** Implemented a state-of-the-art Time Series architecture.
- [x] **Probabilistic Forecasting:** Output P10, P50, and P90 confidence intervals instead of binary flags.
- [x] **Explainability:** Generated "Attention Weight" maps to show *why* a spike is predicted (e.g., Seasonality vs. Sentinel Risk).

## Phase 11: Network Intelligence (Blueprint 1)
- [x] **Risk Propagation Engine:** Implemented a recursive graph traversal algorithm.
- [x] **Logic:** If a factory node fails (Score 10), risk decays across the graph edges to infect related NDCs and Sister Companies ("Financial Contagion").