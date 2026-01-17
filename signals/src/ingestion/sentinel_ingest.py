from typing import List, Dict, Optional
import feedparser
from bs4 import BeautifulSoup
import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
import polars as pl
from pathlib import Path
from email.utils import parsedate_to_datetime
from datetime import datetime

# Load environment variables
load_dotenv()


def fetch_enforcement_reports(feed_url: Optional[str] = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml") -> List[Dict[str, any]]:
    """
    Fetches FDA MedWatch/Safety Reports with a browser-mimicking User-Agent.
    Includes a 'Circuit Breaker' to return demo data if the live feed fails.
    """
    print(f"📡 Connecting to FDA Feed: {feed_url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*"
    }

    raw_feed_content = None

    try:
        response = requests.get(feed_url, headers=headers, timeout=15)
        response.raise_for_status()
        raw_feed_content = response.content
        print("   ✅ Connection established.")

    except Exception as e:
        print(f"   ⚠️ LIVE FEED ERROR: {e}")
        print("   🔄 Activating Circuit Breaker: Switching to Mock Data.")

        return [
            {
                "title": "[DEMO] Urgent: Sterile Water Contamination", "link": "http://fda.gov/demo/1",
                "published": datetime(2026, 1, 16, 12, 0, 0),
                "summary": "Urgent recall issued for Baxter International sterile water vials due to particulate matter observed in lot #4459. Risk of embolism."
            },
            {
                "title": "[DEMO] Labeling Error: Ibuprofen", "link": "http://fda.gov/demo/2",
                "published": datetime(2026, 1, 15, 9, 30, 0),
                "summary": "Voluntary recall of Ibuprofen 200mg by Dr. Reddy's due to potential missing child-safety cap mechanism. No chemical defects found."
            }
        ]

    feed = feedparser.parse(raw_feed_content)
    extracted_data = []
    for entry in feed.entries:
        summary_html = entry.get('summary', '') or entry.get('description', '')
        soup = BeautifulSoup(summary_html, 'html.parser')
        cleaned_summary = soup.get_text()

        try:
            raw_date = entry.get('published')
            dt_object = parsedate_to_datetime(
                raw_date) if raw_date else datetime.now()
        except Exception:
            dt_object = datetime.now()

        extracted_data.append({
            'title': entry.get('title', 'N/A'), 'link': entry.get('link', 'N/A'),
            'published': dt_object, 'summary': cleaned_summary.strip()
        })

    print(f"   ✅ Fetched {len(extracted_data)} reports from live feed.")
    return extracted_data


def analyze_risk_with_gemini(text_batch: List[str]) -> List[Dict[str, any]]:
    """
    Analyzes a batch of text for supply chain risks using Google GenAI (v1.0+ SDK).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # --- UPDATED: New Client Syntax ---
    client = genai.Client(api_key=api_key)

    system_prompt = """
    You are a supply chain risk analyst. For each news summary provided, perform the following tasks:
    1. Identify if the text mentions: 'Factory Shutdown', 'Recall', 'Form 483 Warning', or 'Quality Control Failure'. If none, use 'No Specific Risk Identified'.
    2. Extract the 'Manufacturer Name' (or null).
    3. Extract the 'Drug/Product Name' (or null).
    4. Assign a 'Severity Score' from 0 (No Risk) to 10 (Critical Failure).

    Output must be a JSON object with a single key "analyses" containing a list of objects.
    Each object must have: 'risk_type', 'manufacturer', 'product', 'severity_score'.
    """
    summaries_for_prompt = "\n".join(
        [f"<summary>{s}</summary>" for s in text_batch])
    full_prompt = f"{system_prompt}\n\nSummaries to analyze:\n{summaries_for_prompt}"

    retries = 3
    while retries > 0:
        try:
            print("   🧠 Thinking (Gemini)...")

            # --- UPDATED: New Generation Call ---
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json'
                )
            )

            analysis_result = json.loads(response.text)
            return analysis_result.get("analyses", [])
        except Exception as e:
            print(f"   ⚠️ API Error: {e}. Retrying...")
            retries -= 1
            time.sleep(2)

    return [{"risk_type": "Error", "manufacturer": "Unknown", "severity_score": 0, "product": "Unknown"} for _ in text_batch]


def fetch_and_score_rss() -> pl.DataFrame:
    """
    Fetches the latest FDA enforcement reports, analyzes them for risk with an LLM,
    and returns the scored events as a Polars DataFrame.
    """
    print("\n🚀 Running Sentinel RSS Fetch & Score...")

    # 1. Fetch
    reports = fetch_enforcement_reports()
    if not reports:
        return pl.DataFrame()

    reports_with_summaries = [r for r in reports if r.get('summary')]
    if not reports_with_summaries:
        return pl.DataFrame()

    # 2. Analyze
    summaries = [report['summary'] for report in reports_with_summaries]
    print(f"   🔍 Analyzing {len(summaries)} summaries...")
    analyses = analyze_risk_with_gemini(summaries)

    # 3. Merge
    combined_data = []
    limit = min(len(reports_with_summaries), len(analyses))

    for i in range(limit):
        report = reports_with_summaries[i]
        analysis = analyses[i]
        combined_data.append({
            "title": report.get('title'),
            "link": report.get('link'),
            "event_date": report.get('published'),
            "manufacturer": analysis.get('manufacturer'),
            "risk_type": analysis.get('risk_type'),
            "severity_score": analysis.get('severity_score'),
            "raw_summary": report.get('summary')
        })

    if not combined_data:
        return pl.DataFrame()

    return pl.DataFrame(combined_data)


def save_scored_risks(df: pl.DataFrame, output_path: str = "data/processed/sentinel_risks.parquet"):
    """
    Saves the scored risk DataFrame to a parquet file.
    """
    if df.is_empty():
        print("   ⚠️ No data to save.")
        return

    print(f"   💾 Saving {len(df)} risk events to {output_path}...")
    output_dir = Path(os.path.dirname(output_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = df.select([
        pl.col("event_date").cast(pl.Date), "manufacturer", "risk_type",
        "severity_score", "raw_summary"
    ])
    final_df.write_parquet(output_path)
    print(f"   ✅ SUCCESS: Saved data.")


if __name__ == '__main__':
    # Preserves the original script behavior when run directly
    scored_events_df = fetch_and_score_rss()
    if not scored_events_df.is_empty():
        save_scored_risks(scored_events_df)
