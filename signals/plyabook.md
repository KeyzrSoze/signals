# 📘 Signals: Operational Playbook

**Objective:** This document outlines the step-by-step execution path to run the "Signals" Early Warning System, from raw data ingestion to financial risk simulation.

**Frequency:** Recommended execution is **Weekly** (e.g., Wednesday mornings), after CMS updates the NADAC pricing files.

---

## 🛠 Phase 0: Prerequisites & Infrastructure

Before running the pipelines, ensure your environment is active and the database is listening.

1.  **Activate Environment**
    ```bash
    conda activate signals
    ```

2.  **Start Graph Database**
    Launch the Neo4j container.
    ```bash
    docker-compose up -d
    ```
    * **Verify:** Open [http://localhost:7474](http://localhost:7474).
    * **Login:** `neo4j` / `password` (or the password you set in `docker-compose.yml`).

---

## 📡 Phase 1: The Ingestion Engine (Raw Data)

**Goal:** Fetch the latest pricing, shortage, and risk data from government sources.

1.  **Fetch Pricing History (CMS NADAC)**
    * **Action:** Downloads the latest weekly pricing files from Medicaid.gov.
    * **Command:**
        ```bash
        python src/ingestion/nadac_ingest.py
        ```

2.  **Scrape FDA Shortages**
    * **Action:** Pulls the current active shortage list and updates the event history.
    * **Command:**
        ```bash
        python src/ingestion/fda_shortages.py
        ```

3.  **Run the Sentinel (Risk Intelligence)**
    * **Action:** Hits FDA RSS feeds/APIs for Warning Letters and uses Gemini to score them (0-10).
    * **Command:**
        ```bash
        python src/ingestion/sentinel_ingest.py
        ```

---

## 🕸 Phase 2: The Knowledge Graph (Structure)

**Goal:** Update the physical map of the supply chain and propagate new risks.

1.  **Hydrate the Graph**
    * **Action:** Loads the latest `ndc_entity_map` into Neo4j nodes (Corporations, NDCs, Ingredients).
    * **Command:**
        ```bash
        python src/graph/hydrate_baseline.py
        ```

2.  **Link Facilities (The "Detective")**
    * **Action:** Uses Gemini to fuzzy-match Manufacturer names to physical FDA Facility addresses (FEI numbers).
    * **Command:**
        ```bash
        python src/graph/enrich_facilities.py
        ```

3.  **Propagate Risk (The "Contagion")**
    * **Action:** If Sentinel found a high-risk event (e.g., Factory Failure), this script pushes that risk down the graph edges to infect related NDCs.
    * **Command:**
        ```bash
        # Note: This is typically called programmatically, but can be forced for specific events:
        python -c "from src.graph.risk_engine import RiskEngine; RiskEngine().propagate_factory_failure(fei_number='[TARGET_FEI]', severity_score=10)"
        ```

---

## 🧬 Phase 3: Feature Engineering (The Merge)

**Goal:** Create a unified "Time Machine" dataset for the AI.

1.  **Extract Graph Embeddings**
    * **Action:** Calculates "Supplier Diversity Scores" and topological vectors for every drug.
    * **Command:**
        ```bash
        python src/features/extract_graph_embeddings.py
        ```

2.  **Generate Master Signals**
    * **Action:** Merges Price History (NADAC) + Sentinel Risks + Graph Embeddings using `join_asof` to prevent data leakage.
    * **Command:**
        ```bash
        python src/features/signal_generator.py
        ```
    * **Output:** `data/processed/weekly_features.parquet`

---

## 🧠 Phase 4: AI Modeling (The Brain)

**Goal:** Retrain the Deep Learning model on the latest data.

1.  **Train Temporal Fusion Transformer (TFT)**
    * **Action:** Trains the probabilistic forecasting model.
    * **Command:**
        ```bash
        python src/models/train_tft.py
        ```
    * **Artifact:** Saves best model to `src/models/artifacts/tft_model.ckpt`.

2.  **Generate Explanations (XAI)**
    * **Action:** Creates "Attention Weight" plots to show *why* the model is predicting spikes.
    * **Command:**
        ```bash
        python src/reporting/explain_tft.py
        ```
    * **Output:** `reports/tft_explanation.png`

---

## 💸 Phase 5: Financial Simulation (The Value)

**Goal:** Translate AI probabilities into CFO-ready financial exposure.

1.  **Run Shadow Formulary Simulation**
    * **Action:** Uses the TFT's P90 (Worst Case) forecast to simulate financial loss across 1,000 scenarios per drug.
    * **Command:**
        ```bash
        python src/simulation/monte_carlo.py
        ```
    * **Output:** Prints Total Portfolio VaR (Value at Risk) to console/logs.

---

## 🤖 Automation Strategy (Optional)

To run this automatically every Wednesday at 6:00 AM, add this to your `crontab`:

```bash
0 6 * * 3 cd /path/to/signals && /path/to/conda/python src/pipeline_runner.py >> logs/weekly_run.log 2>&1

🤖 Phase 6: Autonomous Mode (Production)

**Goal:** Run the system as a "Set and Forget" service.

### 1. The Architecture
* **The Clock (Celery Beat):** Dictates the schedule (Hourly Watchdog, Weekly Training).
* **The Muscle (Celery Worker):** Executes the Python code.
* **The Memory (Redis):** Stores task queues and execution history.

### 2. Starting the System
You can start the entire nervous system with a single command:

```bash
# This starts the Worker and the Beat scheduler together
sh start_worker.sh