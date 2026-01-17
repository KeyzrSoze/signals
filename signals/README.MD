# Project Signals: Master Canvas

**Project Goal:** Construct an enterprise-grade Early Warning System (EWS) for pharmacy price spikes using "Network Science" & Event Sourcing.
**Core Metric:** Accuracy (Zero Data Leakage).
**Tech Stack:** Python 3.12, Polars, XGBoost, **Neo4j (Graph DB)**, Gemini 2.5 Flash.

## 🛠 Data Source Manifest
| Data Source | Type | Usage | Ingestion Script |
|:---|:---|:---|:---|
| **CMS NADAC** | Structured (CSV) | Pricing History & Baselines | `src/ingestion/nadac_ingest.py` |
| **FDA NDC Directory** | Structured (JSON) | Product Metadata & Labeler Codes | `src/ingestion/ndc_library.py` |
| **FDA Shortages** | Semi-Structured (Web) | Historical Shortage Events | `src/ingestion/fda_shortages.py` |
| **FDA Enforcement** | Unstructured (RSS) | Recalls & Warning Letters | `src/ingestion/sentinel_ingest.py` |
| **FDA FEI Database** | Structured (CSV) | Factory Locations & Registrations | `src/graph/enrich_facilities.py` |

## Phase 1: Environment & Architecture
- [x] Structure: Domain-driven directory tree.
- [x] Environment: Conda environment `signals` active.

## Phase 2: The Ingestion Engine
- [x] **2A: NADAC Processor:** Strict schema enforcement & case-insensitive normalization.
- [x] **2B: FDA Shortage Scraper:** Longitudinal "Event Stream" (Start/Update/End).
- [x] **2C: Reference Data:** Master list of valid NDCs.

## Phase 3: The Entity Map (Graph Upgrade)
**Objective:** Move from Flat Files to a Knowledge Graph.
- [x] **3A: Graph Hydration:** Ported Entity Map to Neo4j.
- [x] **3B: Facility Resolution:** Used Gemini 2.5 Flash to link "Corporate Subsidiaries" to "Physical Factories" (FEI Numbers).
- [x] **3C: Risk Propagation:** Implemented "Contagion Logic" where factory risks flow to NDCs.

## Phase 4: Signal Generation
- [x] **Time Machine Logic:** `join_asof` for point-in-time correctness.
- [x] **Graph Embeddings:** Extracted Node2Vec structural features.
- [x] **Market Structure:** Calculated HHI (Monopoly Index).

## Phase 5: Modeling (XGBoost)
- [x] **Model Architecture:** XGBoost Classifier trained on `graph_features` + `market_signals`.
- [x] **Performance:** Recall improved by detecting "Hidden Risks" (Sister Company contagion).

## Phase 6: Consulting Deliverables
- [x] **Automated Risk Reporting:** "Monday Morning" watchlist with "Reason Codes" (e.g., "Upstream Factory Failure").

## Phase 7: The Sentinel (Unstructured Intelligence)
- [x] **LLM "Signal Transducer":** Gemini Flash parses FDA text into quantitative `severity_scores` (0-10).

## Phase 8: Financial Risk Simulation
- [x] **Monte Carlo Engine:** Simulates 1,000 market scenarios to calculate Value at Risk (VaR).