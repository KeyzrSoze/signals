from typing import List, Dict, Optional
import feedparser
from bs4 import BeautifulSoup
import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def fetch_enforcement_reports(feed_url: Optional[str] = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/enforcement-reports/rss.xml") -> List[Dict[str, str]]:
    """
    Fetches and processes FDA Enforcement Report RSS feed entries.

    Args:
        feed_url: The URL of the RSS feed. Defaults to the official FDA feed.
                  Can be a local file path for testing.

    Returns:
        A list of dictionaries, where each dictionary contains the title,
        link, published date, and cleaned summary of an enforcement report.
    """
    if not feed_url:
        raise ValueError("feed_url cannot be None or empty.")

    print(f"Fetching RSS feed from: {feed_url}")
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"Error parsing feed from {feed_url}: {e}")
        return []

    if feed.bozo:
        print(
            f"Warning: Feed from {feed_url} may be ill-formed. Bozo reason: {feed.bozo_exception}")

    extracted_data = []
    for entry in feed.entries:
        summary_html = entry.get('summary', '')
        # Use BeautifulSoup to strip HTML tags from the summary
        soup = BeautifulSoup(summary_html, 'html.parser')
        cleaned_summary = soup.get_text()

        extracted_data.append({
            'title': entry.get('title', 'N/A'),
            'link': entry.get('link', 'N/A'),
            'published': entry.get('published', 'N/A'),
            'summary': cleaned_summary.strip()
        })
    print(f"Fetched {len(extracted_data)} reports.")
    return extracted_data


def analyze_risk_with_gemini(text_batch: List[str]) -> List[Dict[str, any]]:
    """
    Analyzes a batch of text for supply chain risks using the Gemini API.

    Args:
        text_batch: A list of strings, where each string is a summary to analyze.

    Returns:
        A list of dictionaries containing the structured risk analysis.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    system_prompt = """
You are a supply chain risk analyst. For each news summary provided, perform the following tasks:
1.  Identify if the text mentions one of the following risk types: 'Factory Shutdown', 'Recall', 'Form 483 Warning', or 'Quality Control Failure'. If none are mentioned, the value should be 'No Specific Risk Identified'.
2.  Extract the 'Manufacturer Name' if mentioned. If not, this should be null.
3.  Extract the 'Drug/Product Name' if mentioned. If not, this should be null.
4.  Assign a 'Severity Score' from 0 (No Risk) to 10 (Critical Failure).

You MUST return a JSON object with a single key "analyses" which contains a list of JSON objects. Each object in the list corresponds to a summary in the input and must contain the following keys: 'risk_type', 'manufacturer', 'product', 'severity_score'.
The number of objects in the "analyses" list must be equal to the number of summaries provided.
Your response must be only the JSON object, with no other text or markdown.
"""

    # Create a single prompt with all summaries
    summaries_for_prompt = "\n".join(
        [f"<summary>{summary}</summary>" for summary in text_batch])

    prompt = f"{system_prompt}\n\nHere are the summaries to analyze:\n{summaries_for_prompt}"

    retries = 3
    while retries > 0:
        try:
            print("Sending request to Gemini API...")
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"))
            response_text = response.text.strip()

            # The model is configured to return JSON, so we can parse it directly.
            analysis_result = json.loads(response_text)
            print("Received and parsed response from Gemini.")
            return analysis_result.get("analyses", [])
        except Exception as e:
            print(f"An error occurred with the Gemini API: {e}. Retrying...")
            retries -= 1
            time.sleep(5)  # Wait 5 seconds before retrying

    print("Failed to analyze text after multiple retries.")
    return [{"error": "Failed to analyze", "original_text": text} for text in text_batch]


def run_sentinel():
    """
    Runs the full Sentinel pipeline: fetch, analyze, structure, and save.
    """
    print("Starting Sentinel risk analysis pipeline...")

    # 1. Fetch raw data
    reports = fetch_enforcement_reports()
    if not reports:
        print("No reports fetched. Exiting pipeline.")
        return

    # Filter out reports with no summary to analyze
    reports_with_summaries = [r for r in reports if r.get('summary')]
    if not reports_with_summaries:
        print("No summaries found in fetched reports. Exiting pipeline.")
        return

    summaries = [report['summary'] for report in reports_with_summaries]

    # 2. Analyze with Gemini
    print(f"Analyzing {len(summaries)} summaries for supply chain risks...")
    analyses = analyze_risk_with_gemini(summaries)

    if not analyses or any("error" in an for an in analyses):
        print("Risk analysis failed or returned errors. Exiting pipeline.")
        return

    # 3. Combine and structure data
    combined_data = []
    for report, analysis in zip(reports_with_summaries, analyses):
        # The 'product' from analysis is not in the final schema, so we omit it.
        combined_data.append({
            "event_date": report.get('published'),
            "manufacturer": analysis.get('manufacturer'),
            "risk_type": analysis.get('risk_type'),
            "severity_score": analysis.get('severity_score'),
            "raw_summary": report.get('summary')
        })

    if not combined_data:
        print("No data was successfully combined. Exiting pipeline.")
        return

    # 4. Convert to Polars DataFrame and clean
    print("Converting to Polars DataFrame...")
    df = pl.DataFrame(combined_data)

    # Cast date column. Polars' `to_datetime` can often infer formats automatically.
    # Using `strict=False` makes it robust to slight variations in RSS date format.
    df = df.with_columns(
        pl.col("event_date").str.to_datetime(strict=False).dt.date()
    )

    # 5. Ensure final schema as per requirements
    final_df = df.select([
        "event_date",
        "manufacturer",
        "risk_type",
        "severity_score",
        "raw_summary"
    ])

    # 6. Save to Parquet
    # This path is relative to the project root where the script is run.
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sentinel_risks.parquet"

    final_df.write_parquet(output_path)
    print(f"Successfully saved {len(final_df)} records to {output_path}")


if __name__ == '__main__':
    run_sentinel()
