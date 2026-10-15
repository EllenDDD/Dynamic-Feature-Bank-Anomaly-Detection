"""Optional GPT semantic-consistency interface.

The validator is intentionally disabled by default.  Keeping the provider
behind this small interface lets experiments enable an external API later,
while normal training and inference remain fully offline and deterministic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen


@dataclass
class GPTValidationResult:
    text: str
    score: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    enabled: bool = False


class GPTSemanticValidator:
    """Optional API-backed validator; no request is made unless enabled=True."""

    def __init__(
        self,
        enabled: bool = False,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-5.5",
        api_key: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = float(timeout)

    def validate(self, text: str, category: str = "texture") -> GPTValidationResult:
        """Validate a description, returning the input unchanged when disabled."""
        if not self.enabled:
            return GPTValidationResult(text=text, enabled=False)
        if not self.api_key:
            raise RuntimeError("GPT semantic validation is enabled but OPENAI_API_KEY is missing.")
        prompt = (
            "Evaluate whether the following industrial-texture description is "
            "consistent, category-relevant, and useful for anomaly detection. "
            "Return JSON with keys score (0..1) and revised_text.\n"
            f"Category: {category}\nDescription: {text}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        revised = str(parsed.get("revised_text", text))
        score = parsed.get("score")
        return GPTValidationResult(text=revised, score=float(score) if score is not None else None, raw=result, enabled=True)

