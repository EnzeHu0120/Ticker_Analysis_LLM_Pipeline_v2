from __future__ import annotations

"""
Minimal sanity check for Vertex AI + Gemini 3.1 Pro Preview using ADC.

Prerequisites (one-time on this machine):
  gcloud auth application-default login

And in your environment (e.g. .env or shell):
  GOOGLE_CLOUD_PROJECT=ticker-analysis-pipeline
  GOOGLE_CLOUD_LOCATION=global

Then run:
  python test_vertex.py
"""

from google import genai


def main() -> None:
    client = genai.Client(
        vertexai=True,
        project="ticker-analysis-pipeline",
        location="global",
    )

    resp = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents="Return a short JSON object with keys message and status.",
    )
    print(resp.text)


if __name__ == "__main__":
    main()

