# Project Signals: Master Canvas
Project Goal: Construct an enterprise-grade Early Warning System (EWS) for pharmacy price spikes using "Event Sourcing" architecture. Core Metric: Accuracy (Zero Data Leakage). Tech Stack: Python 3.12 (Conda), Polars, XGBoost, Git.
- ## Phase 1: Environment & Architecture (Completed)
    - [x] Structure: Domain-driven directory tree (src/ingestion, src/models, etc.).
    - [x] Environment: Conda environment signals active.
    - [x] Version Control: Git initialized with .gitignore for data security

- ## Phase 2: The Ingestion Engine (Current Focus)
    Objective: Build automated, resilient data pipelines.
    - 2A: NADAC Processor (Today): * Action: Fetch raw NADAC weekly files.
        - Logic: Enforce strict schema (NDC=String), handle date parsing, validation.
    - 2B: FDA Shortage Scraper: * Action: Parse FDA API history.
         - Logic: Convert "current status" into a longitudinal "Event Stream" (Start/Update/End).
    - 2C: Reference Data: * Action: Ingest FDA NDC Directory.
        - Logic: Build the master list of valid NDCs.
     - ## Phase 3: The Entity Map (The "Rosetta Stone")
        Objective: Link disjointed data sources.
        - Action: Create a bridge table: NDC (11-digit) $\leftrightarrow$ Ingredient $\leftrightarrow$ Manufacturer
        - Critical Logic: Fuzzy matching for manufacturer names (e.g., "Teva" vs. "Teva Pharm") to ensure signals don't get lost.
    - ## Phase 4: Signal Generation (Feature Engineering)
        Objective: Create the training dataset weekly_market_signals
        - Time Machine Logic: Use Polars join_asof to align price data (Mondays) with the latest known shortage status from the past.
        - Features:
            - Shortage Active (True/False)
            - Supply Concentration (HHI index)
            - Price Velocity (First Derivative of Price)
    - ## Phase 5: Modeling (The "Hurdle" Architecture)
        Objective: Train high-precision models.
        - Model 1 (The Sentry): XGBoost Classifier (Will it spike?).
        - Model 2 (The Quant): LightGBM Regressor (By how much?)
        - Validation: Walk-Forward Backtesting (Train 2021-2022, Test 2023).
    - ## Phase 6: Consulting Deliverables
        Objective: Client-facing assets.
        - Outputs: Automated "Watchlist" Reports (Excel/PDF), Slide Deck generator for high-risk drugs.
        
        