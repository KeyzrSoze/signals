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

---

## 🚨 Phase 6: The Watchdog (Active Monitoring)

**Goal:** Real-time alerting for immediate threats (Factory Failures, FDA Warnings).

1.  **The Sentinel**
    * **Action:** Runs hourly to scan FDA RSS feeds.
    * **Trigger:** Automated via Celery Beat (or manual override).
    * **Command:** `python src/tasks/sentinel_tasks.py`
    * **Output:** Pushes Critical Alerts (> Score 8) to Slack/Teams immediately.

---

## ⚖️ Phase 7: The Scorecard (Self-Evaluation)

**Goal:** Auditing the AI's performance against reality.

1.  **The Reconciliation Loop**
    * **Action:** Compares last month's predictions against today's actual NADAC prices.
    * **Command:** `python src/evaluation/scorecard.py`
    * **Output:** Updates `data/outputs/prediction_registry.parquet` and generates `reports/model_trust_score.png`.


### Starting the System
You can start the entire nervous system with a single command:

```bash
# This starts the Worker and the Beat scheduler together
sh start_worker.sh

This guide covers setup, execution, and crucially, **how to interpret and pitch the deliverables** to business stakeholders.

---

### **Part 1: The Setup (Day 0)**

*Do this once to initialize the system.*

**1. Infrastructure Initialization**

* **Action:** Boot up the brain (Graph DB) and the memory (Redis).
* **Command:** `docker-compose up -d`
* **Verification:**
* Neo4j Browser: `http://localhost:7474` (User: `neo4j` / Pass: `password`)
* Redis: Check logs or ensure container is running.



**2. Environment & Dependencies**

* **Action:** Install the Python "muscles."
* **Command:**
```bash
conda activate signals
pip install -r requirements.txt

```



**3. Graph Hydration (The Map)**

* **Action:** Load your baseline supply chain map (Manufacturers -> Factories -> Drugs).
* **Commands:**
```bash
# 1. Define the Rules (Constraints)
python src/graph/setup_db.py

# 2. Load the Map (Nodes & Edges)
python src/graph/hydrate_baseline.py

# 3. Connect the Factories (The "Detective" Layer)
python src/graph/enrich_facilities.py

```

---

### **Part 2: The Routine (Weekly Execution)**

*How to run the system in production. You can run these manually or use the `start_worker.sh` for autonomous mode.*

**Step 1: The Ingestion (Wake Up)**

* **Goal:** Get the latest prices and news.
* **Scripts:**
* `src/ingestion/nadac_ingest.py` (Prices)
* `src/ingestion/sentinel_ingest.py` (Risk News)
* `src/ingestion/fda_shortages.py` (Shortage Status)



**Step 2: The "Brain" Update (Retrain)**

* **Goal:** Teach the AI about the new data from the last week.
* **Scripts:**
* `src/features/extract_graph_embeddings.py` (Update Supply Chain scores)
* `src/features/signal_generator.py` (Merge all data)
* `src/models/train_tft.py` (Train the Deep Learning Model)



**Step 3: The Forecast (Predict)**

* **Goal:** Generate the forward-looking predictions.
* **Scripts:**
* `src/reporting/explain_tft.py` (Generate "Why" plots)
* `src/simulation/monte_carlo.py` (Calculate Financial Loss)
* `src/reporting/generate_watchlist.py` (Create the CSV report)



---

### **Part 3: The Deliverables & How to Explain Them**

*This is the most critical part: Translating Python output into Business Value.*

#### **Deliverable A: The "Monday Morning" Watchlist**

* **File:** `reports/weekly_watchlist.csv`
* **What it is:** A prioritized list of drugs likely to spike in price in the next 8 weeks.
* **How to Explain it (The Pitch):**
> "This isn't just a list of shortages. It's a list of **Inflation Risks**.
> Look at **Amoxicillin**: The system flagged it not because the price moved (it hasn't yet), but because a factory in India just failed inspection (Sentinel Score 9) and import volumes dropped.
> **Recommendation:** We should buy 3 months of inventory *now* at the current price ($5.00) before it spikes to $12.00 next month."



#### **Deliverable B: The "Risk Tunnel" Plot**

* **File:** `reports/tft_explanation.png`
* **What it is:** A chart showing the historical price and the AI's predicted price range (P10 to P90).
* **How to Explain it:**
> "See this shaded tunnel? That is the AI's confidence interval.
> The 'Actual' price is the solid line. The AI is predicting a **90% probability** (the top of the tunnel) that the price will breach $15.00 by March.
> The 'Feature Importance' bar chart below shows *why*: 60% of this prediction is driven by the **Supplier Risk Score**, not historical seasonality."



#### **Deliverable C: The Financial "Value at Risk" (VaR)**

* **Output:** Console Log / Report Summary from `monte_carlo.py`
* **What it is:** A dollar amount representing potential loss (e.g., "Portfolio VaR: $450,000").
* **How to Explain it (To the CFO):**
> "We ran 1,000 simulations on our current portfolio.
> In the **Worst Case Scenario** (95th percentile), we stand to lose **$450,000** to unexpected price hikes this quarter.
> This represents our 'Unhedged Inflation Exposure.' We can reduce this to $50,000 if we lock in contracts for the top 5 drugs on the Watchlist today."



---

### **Part 4: The Autonomous Mode (Blueprint 4)**

*If you want to "Set and Forget" the system.*

1. **Start the Worker:**
Run `sh start_worker.sh` in a terminal (or background process).
2. **What Happens:**
* **Hourly:** The system checks FDA feeds. If a "Factory Shutdown" (Score 10) is detected, it alerts you *immediately*.
* **Weekly (Wed 06:00):** It automatically runs the full pipeline (Part 2) and updates the deliverables.



### **Summary Checklist for Success**

1. [ ] **Graph Hydrated?** (Check Neo4j for nodes)
2. [ ] **Prices Fresh?** (Check `data/raw/nadac` for this week's file)
3. [ ] **Model Trained?** (Check date on `src/models/artifacts/tft_model.ckpt`)
4. [ ] **Story Ready?** (Have the Watchlist and VaR numbers ready for your meeting)