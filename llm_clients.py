from __future__ import annotations

"""
llm_clients.py

Thin LLM JSON runners so orchestration can stay focused on data flow.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

try:
    from google.genai import types
except ImportError:  # pragma: no cover - environment-dependent
    types = None  # type: ignore[assignment]


def extract_json(text: str) -> Any:
    payload = (text or "").strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class GeminiRunner:
    model: str
    client: Any
    system_prompt: str

    def run_json(self, prompt: str, temperature: float = 0.1) -> Any:
        if types is None:
            raise RuntimeError("google-genai package is required for GeminiRunner")
        cfg = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type="application/json",
            temperature=temperature,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=cfg,
        )
        return extract_json(resp.text)


@dataclass
class OpenAIRunner:
    model: str
    client: Any
    system_prompt: str

    def run_json(self, prompt: str, temperature: float = 0.1) -> Any:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        else:
            text = str(content or "")
        return extract_json(text)
