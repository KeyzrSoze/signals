# debug_models.py
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def list_available_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)

    print("🔍 Scanning for available Gemini models...")
    try:
        # Pager through the list of models
        for model in client.models.list():
            # Filter for models that can generate content (chat/text)
            if "generateContent" in (model.supported_actions or []):
                print(f"   ✅ Found: {model.name}")

    except Exception as e:
        print(f"   ❌ Error listing models: {e}")


if __name__ == "__main__":
    list_available_models()
