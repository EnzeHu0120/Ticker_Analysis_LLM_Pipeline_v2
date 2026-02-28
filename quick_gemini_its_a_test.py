"""Quick sanity check: Gemini API connectivity. Uses .env for GEMINI_API_KEY (no keys in repo)."""
from pathlib import Path
import os

# Load .env from project root so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from google import genai

client = genai.Client()  # reads GEMINI_API_KEY from env or .env
resp = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Say hello in Chinese in one sentence."
)
print(resp.text)
