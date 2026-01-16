# Project Signals: Master Canvas

**Project Goal:** Construct an enterprise-grade Early Warning System (EWS) for pharmacy price spikes using "Event Sourcing" architecture.
**Core Metric:** Accuracy (Zero Data Leakage).
**Tech Stack:** Python 3.12 (Conda), Polars, XGBoost, Git.

## Phase 1: Environment & Architecture
- [x] Structure: Domain-driven directory tree (src/ingestion, src/models, etc.).
- [x] Environment: Conda environment `signals` active.
- [x] Version Control: Git initialized with .gitignore for data security.

## Phase 2: The Ingestion Engine
**Objective:** Build automated, resilient data pipelines.
- [x] **2A: NADAC Processor**
    - Action: Fetch raw NADAC weekly files.
    - Logic: Enforce strict schema (NDC=String), implemented "Case-Insensitive" column normalization to handle multi-year government format changes.
- [x] **2B: FDA Shortage Scraper**
    - Action: Parse FDA API history.
    - Logic: Converted "current status" snapshot into a longitudinal "Event Stream" (Start/Update/End) with robust date parsing for various FDA formats.
- [x] **2C: Reference Data**
    - Action: Ingest FDA NDC Directory.
    - Logic: Built the master list of valid NDCs and product metadata.

## Phase 3: The Entity Map (The "Rosetta Stone")
**Objective:** Link disjointed data sources.
- [x] **Entity Map Construction**
    - Action: Created a bridge table: NDC (11-digit) <-> Ingredient <-> Manufacturer.
    - Logic: Implemented "Smart Fallback" logic.
        - Direct Match: 9-digit Product Code.
        - Fallback 1: Manufacturer inference via 5-digit Labeler Code.
        - Fallback 2: Ingredient extraction via text mining of Drug Description.
    - Result: Achieved >85% manufacturer coverage and 100% ingredient coverage.

## Phase 4: Signal Generation (Feature Engineering)
**Objective:** Create the training dataset `weekly_market_signals` and `weekly_features`.
- [x] **Time Machine Logic**
    - Implemented Polars `join_asof` to align price data (Mondays) with the latest known shortage status from the past to prevent data leakage.
- [x] **Advanced Feature Engineering ("Kitchen Sink")**
    - **Shortage Signals:** Active status, duration (weeks in shortage).
    - **Market Structure:** Calculated HHI (Herfindahl-Hirschman Index) to detect monopolies and competitor counts.
    - **Price Dynamics:** Calculated 4-week Price Velocity (Momentum) and 12-week Volatility (Coefficient of Variation).

## Phase 5: Modeling (The "Hurdle" Architecture)
**Objective:** Train high-precision models.
- [x] **Model Architecture**
    - Implementation: XGBoost Classifier.
    - Target: Binary classification of >10% price spike within an 8-week lookahead window.
- [x] **Training Strategy**
    - Logic: Time-Based Split (Train on historical data, Test on latest year).
    - Optimization: Used `scale_pos_weight` to handle class imbalance (rare spike events).
    - Performance: Achieved ~68% Recall (capturing 2/3rds of all spikes) with high importance placed on Price Velocity and Market HHI.

## Phase 6: Consulting Deliverables
**Objective:** Client-facing assets.
- [x] **Automated Risk Reporting**
    - Action: Script `generate_watchlist.py` generates a CSV report.
    - Output: "Monday Morning" watchlist ranking drugs by `risk_score` (probability of spike), enriched with human-readable descriptions, monopoly indices, and momentum indicators.

## Phase 7: The Sentinel (Unstructured Intelligence)
**Objective:** Predict supply chain failures *before* they hit the structured data.
- [x] **Unstructured Data Ingestion**
    - Action: Built `sentinel_ingest.py` to listen to FDA Enforcement Reports (RSS) and trade news feeds.
    - Logic: Targets "Lagging Indicators" (Price) vs "Leading Indicators" (Factory Inspections).
- [x] **LLM "Signal Transducer"**
    - Implementation: Integrated **Gemini Flash** model via API.
    - Logic: Applied an "LLM Judge" pattern to classify raw text into structured risk events (e.g., "Form 483 Issued", "Voluntary Recall", "Factory Shutdown").
    - Output: Normalized `severity_score` (0-10) mapped to specific Manufacturers.