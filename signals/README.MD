# Project Signals: Master Canvas

**Project Goal:** Construct an enterprise-grade Early Warning System (EWS) for pharmacy price spikes using **Graph Neural Networks** & **Autonomous Agents**.
**Core Metric:** Accuracy (Zero Data Leakage) & Latency (Real-time Detection).
**Tech Stack:**
* **Core:** Python 3.12, Polars, Git.
* **Infrastructure:** Docker, Redis (Caching), Celery (Orchestration).
* **Graph:** Neo4j (Graph Database), Cypher.
* **AI/ML:** PyTorch Forecasting (TFT), XGBoost.
* **GenAI:** Google Gemini 2.5 Flash.

## Phase 1: Environment & Architecture
- [x] Structure: Domain-driven directory tree.
- [x] Environment: Conda environment `signals` active.

## Phase 2: The Ingestion Engine
- [x] **2A: NADAC Processor:** Strict schema enforcement & normalization.
- [x] **2B: FDA Shortage Scraper:** Longitudinal "Event Stream."
- [x] **2C: Reference Data:** Master list of valid NDCs.

## Phase 3: The Entity Map (Graph Upgrade)
- [x] **Entity Map:** Bridge table (NDC <-> Ingredient <-> Manufacturer).
- [x] **Graph Hydration:** Ported flat maps into **Neo4j**.
- [x] **Facility Enrichment:** Linked subsidiaries to physical FDA factories via Gemini.

## Phase 4: Signal Generation
- [x] **Time Machine Logic:** `join_asof` for point-in-time correctness.
- [x] **Graph Topology:** Extracted FastRP Embeddings and Supplier Diversity Scores.

## Phase 5: Baseline Modeling (XGBoost)
- [x] **Binary Classification:** Predicted >10% spikes with ~68% recall.

## Phase 6: Consulting Deliverables
- [x] **Automated Risk Reporting:** "Monday Morning" watchlist generation.

## Phase 7: The Sentinel (Unstructured Intelligence)
- [x] **LLM "Signal Transducer":** Gemini Flash parses FDA text into risk scores.

## Phase 8: Financial Risk Simulation
- [x] **Shadow Formulary:** Monte Carlo simulation for Value at Risk (VaR).

## Phase 9: System Integration
- [x] **Temporal Joining:** Unified Graph features, Price history, and Sentinel scores.

## Phase 10: Deep Learning (Blueprint 3)
- [x] **Temporal Fusion Transformer (TFT):** Probabilistic forecasting (P10/P90).
- [x] **Explainability:** Attention-based feature importance maps.

## Phase 11: Network Intelligence (Blueprint 1)
- [x] **Risk Propagation:** Recursive graph traversal for "Contagion" modeling (Factory -> Drug).

## Phase 12: The Autonomous Nervous System (Blueprint 4)
**Objective:** Transform the system from a "Weekly Script" to an "Always-On Platform."
- [x] **Infrastructure:** Deployed **Redis** (State Memory) and **Celery** (Task Queue).
- [x] **The "Reflex" System:** Implemented an hourly "Watchdog" task that hits FDA feeds and triggers immediate alerts for high-severity events (Score > 8).
- [x] **The "Brain" System:** Scheduled the heavy training pipeline (Ingest -> Train -> Forecast) to run automatically every Wednesday at 06:00 UTC.