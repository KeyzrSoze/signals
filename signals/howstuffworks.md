# Signals: System Architecture & Logic

## 1. The "Elevator Pitch"
**What does this system do?**
"Signals" is a "Digital Twin" of the Pharmaceutical Supply Chain. It doesn't just track prices; it maps the physical web of factories, parent companies, and shipping lanes to predict failures before they happen.

**The "Crystal Ball" Logic:**
1.  **The Ear:** We listen to FDA signals (Recalls, Inspections).
2.  **The Brain (Graph):** A Knowledge Graph (Neo4j) traces that signal from a factory in India to every specific pill bottle on a US pharmacy shelf.
3.  **The Prediction (TFT):** A Deep Learning model (Transformer) combines this "Structural Risk" with "Price Momentum" to forecast exact price ranges 8 weeks out.

---

## 2. The Engineering Deep Dive

### A. The Knowledge Graph (The "Map")
* **Technology:** Neo4j + Gemini 2.5 Flash.
* **The Problem:** Pharma data is opaque. "Teva USA" and "Teva Israel" look like different companies in a database, but they share risk.
* **The Solution:** We built a Graph Topology:
    * `(Factory)-[:MAKES]->(Ingredient)-[:IN]->(Drug)`
    * `(Parent Corp)-[:OWNS]->(Subsidiary)`
* **Risk Propagation:** If a Factory node turns "Red" (Inspection Failure), our **RiskEngine** pushes that risk score down the edges. If you buy a drug connected to that factory, your risk score goes up, even if the price hasn't moved yet.

### B. The Predictive Core (Temporal Fusion Transformer)
* **Architecture:** We replaced standard XGBoost with a **Temporal Fusion Transformer (TFT)**.
* **Why TFT?**
    1.  **Sequence Aware:** It understands that "Winter" follows "Autumn" (Crucial for flu-season drugs like Amoxicillin).
    2.  **Probabilistic:** Instead of saying "Yes/No Spike", it says "90% chance price will be between $5.00 and $7.50".
    3.  **Explainable:** It uses "Attention Heads" to tell us: *"I predicted this spike because the Supplier Diversity Score dropped, not because of price momentum."*

### C. The Shadow Formulary (Financial Simulation)
* **Objective:** Translate abstract probabilities into concrete financial exposure (VaR).
* **Logic:**
    * **Input:** The **P90 Forecast** (Worst Case Scenario) from the TFT model.
    * **Calculation:** `Loss = (P90_Price - Current_Price) * Volume`.
* **Result:** This allows CFOs to budget for "Inflation Shock" rather than just "Risk."

### D. The Sentinel (Unstructured Intelligence)
* **LLM Integration:** Gemini 2.5 Flash acts as a "Signal Transducer," converting messy FDA RSS feeds into structured `severity_scores` (0-10).
* **Graph Enrichment:** Gemini also acts as a "Detective," fuzzy-matching corporate names to physical FDA Facility Establishment Identifiers (FEIs).