import requests
import polars as pl
import os
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "https://api.fda.gov/drug/shortages.json"
PROCESSED_DATA_PATH = "data/processed"


def fetch_all_shortages():
    """
    Paginate through the openFDA API to get ALL shortage records.
    """
    print("🚀 Starting FDA Shortage Ingestion...")
    all_records = []
    skip = 0
    limit = 1000

    while True:
        params = {"limit": limit, "skip": skip}
        try:
            # print(f"   📡 Fetching records {skip} to {skip + limit}...")
            response = requests.get(API_URL, params=params)
            data = response.json()

            if "error" in data:
                break

            results = data.get("results", [])
            if not results:
                break

            all_records.extend(results)
            skip += limit
            time.sleep(0.5)

        except Exception as e:
            print(f"   ❌ Network Error: {e}")
            break

    print(f"   📥 Total Raw Records Fetched: {len(all_records)}")
    return all_records


def process_shortages(raw_records):
    """
    Converts raw FDA records into an 'Event Stream'.
    """
    if not raw_records:
        return None

    print("   ⚙️  Processing Event Stream...")

    events = []

    for r in raw_records:
        # 1. Safe Extraction
        generic_name = r.get("generic_name", "UNKNOWN").upper()
        company = r.get("company_name", "UNKNOWN").upper()
        reason = r.get("shortage_reason", "UNKNOWN")

        # 2. Extract Date Fields
        # The logs showed format is "MM/DD/YYYY"
        start_date_str = r.get("initial_posting_date")

        # Fallback chain for the "End Date"
        change_date_str = (
            r.get("change_date") or
            r.get("status_change_date") or
            r.get("update_date")  # Added based on your inspection logs
        )

        status = r.get("status", "Current")

        # 3. Create "Shortage Start" Event
        if start_date_str:
            events.append({
                "event_date": start_date_str,
                "event_type": "shortage_start",
                "generic_name": generic_name,
                "company_name": company,
                "reason": reason,
                "status_at_event": "Active"
            })

        # 4. Create "Shortage Resolved" Event
        # Only create this if we know for sure it's resolved
        if status == "Resolved" and change_date_str:
            events.append({
                "event_date": change_date_str,
                "event_type": "shortage_resolved",
                "generic_name": generic_name,
                "company_name": company,
                "reason": reason,
                "status_at_event": "Resolved"
            })

    if not events:
        print("   ⚠️ No events generated.")
        return None

    # 5. Robust Date Parsing (Polars)
    # We now handle MM/DD/YYYY (US standard) specifically
    df = pl.DataFrame(events)

    df = (
        df
        .with_columns(
            pl.col("event_date")
              .str.strptime(pl.Date, "%m/%d/%Y", strict=False)  # The fix!
        )
        .filter(pl.col("event_date").is_not_null())
        .unique()
        .sort("event_date")
    )

    return df


def run_pipeline():
    raw_data = fetch_all_shortages()
    if not raw_data:
        return

    df = process_shortages(raw_data)

    if df is not None:
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        output_path = os.path.join(
            PROCESSED_DATA_PATH, "shortage_events.parquet")
        df.write_parquet(output_path)

        print(f"   ✅ SUCCESS! Processed {df.height} historical events.")
        if df.height > 0:
            print(
                f"   📅 Date Range: {df['event_date'].min()} to {df['event_date'].max()}")
        print(f"   💾 Saved to: {output_path}")


if __name__ == "__main__":
    run_pipeline()
