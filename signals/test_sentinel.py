# test_sentinel.py
from src.ingestion.sentinel_ingest import analyze_risk_with_gemini
import os
import sys
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def test_intelligence():
    print("🧪 Starting Sentinel Intelligence Test...")

    # 1. Create FAKE news snippets (One disaster, one boring update)
    fake_news = [
        "URGENT: FDA issues Form 483 to Pfizer after inspecting the Kalamazoo sterile injectable plant. Inspectors found mold in the air filtration system.",
        "AstraZeneca reports Q3 earnings are up 5%. The CEO will host a town hall next Tuesday."
    ]

    print(f"   📤 Sending {len(fake_news)} mock scenarios to Gemini...")

    # 2. Run the actual function
    try:
        results = analyze_risk_with_gemini(fake_news)

        # 3. Validate results
        print("\n   🤖 AI Analysis Results:")
        for res in results:
            print(
                f"      - Risk: {res.get('risk_type')} | Score: {res.get('severity_score')} | Mfg: {res.get('manufacturer')}")

        # 4. Assertions (The "Pass/Fail" check)
        high_risk_hit = any(r['severity_score'] > 5 for r in results)
        boring_hit = any(r['risk_type'] ==
                         'No Specific Risk Identified' for r in results)

        if high_risk_hit and boring_hit:
            print(
                "\n   ✅ SUCCESS: Sentinel correctly distinguished between a Crisis and Noise.")
        else:
            print("\n   ❌ FAILURE: Sentinel failed to distinguish risks.")

    except Exception as e:
        print(f"\n   ❌ ERROR: API Call Failed. Details: {e}")
        print("      (Did you set up your .env file correctly?)")


if __name__ == "__main__":
    load_dotenv()
    test_intelligence()
