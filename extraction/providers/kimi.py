"""Moonshot Kimi (OpenAI-compatible) vision extraction."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from config import AI_MODEL, KIMI_MAX_TOKENS, KIMI_TEMPERATURE, KIMI_THINKING, MAX_RETRIES, RETRY_BACKOFF_S
from extraction.images import normalize_extracted_record


try:
    from openai import OpenAI as _OpenAIClient

    KIMI_AVAILABLE = True
except ImportError:
    KIMI_AVAILABLE = False
    _OpenAIClient = None  # type: ignore[assignment,misc]


def _kimi_k2_5_model() -> bool:
    """Return True for kimi-k2.5 family, which has fixed temperature and thinking mode."""
    return AI_MODEL.startswith("kimi-k2")


def _failed_record(last_error: Exception | str | None, answer_fields: list[str]) -> dict:
    err = str(last_error) if last_error is not None else "unknown"
    base: dict = {
        "student_name": "EXTRACTION_ERROR",
        "student_name_confidence": "failed",
        "confidence": "failed",
        "error": err,
    }
    for f in answer_fields:
        base[f] = "?"
        base[f"{f}_confidence"] = "failed"
    return normalize_extracted_record(base, answer_fields)


class KimiProvider:
    @staticmethod
    def create_client() -> Any | None:
        if not KIMI_AVAILABLE:
            print("Warning: OpenAI package not installed. Run: pip install openai")
            return None
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            print("Warning: KIMI_API_KEY not set. Kimi will not be available.")
            return None
        
        # Support custom base URL for different regions (China vs international)
        # Default: https://api.moonshot.cn/v1 (international)
        # For China users, set KIMI_BASE_URL in .env if different
        base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        
        print(f"[INFO] Connecting to Kimi API at: {base_url}")
        print(f"[INFO] Using API key: {api_key[:15]}...")
        
        assert _OpenAIClient is not None
        return _OpenAIClient(api_key=api_key, base_url=base_url)

    def extract(
        self,
        client: Any,
        image_bytes: bytes,
        prompt: str,
        schema: type[BaseModel],
        page_num: int,
        answer_fields: list[str],
    ) -> dict:
        if not KIMI_AVAILABLE or _OpenAIClient is None:
            return _failed_record("openai package not installed", answer_fields)
        if not isinstance(client, _OpenAIClient):
            print("    Error: Kimi model selected but wrong client type")
            return _failed_record("Client type mismatch for Kimi", answer_fields)
        return self._single(client, image_bytes, page_num, prompt, schema, answer_fields)

    def _single(
        self,
        client: Any,
        image_bytes: bytes,
        page_num: int,
        prompt: str,
        schema: type[BaseModel],
        answer_fields: list[str],
    ) -> dict:
        last_error: Exception | None = None
        backoff = RETRY_BACKOFF_S
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # kimi-k2.5 has fixed temperature (1.0 thinking / 0.6 non-thinking);
        # passing any other value raises a 400 error.
        # For older moonshot-v1-* models, pass the configured temperature normally.
        is_k2_5 = _kimi_k2_5_model()
        extra: dict = {}
        if is_k2_5:
            thinking_type = "enabled" if KIMI_THINKING else "disabled"
            extra["extra_body"] = {"thinking": {"type": thinking_type}}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs: dict = dict(
                    model=AI_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=KIMI_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    **extra,
                )
                if not is_k2_5:
                    kwargs["temperature"] = KIMI_TEMPERATURE
                response = client.chat.completions.create(**kwargs)

                raw = response.choices[0].message.content or ""
                try:
                    data = json.loads(raw)
                    try:
                        schema.model_validate(data)
                    except ValidationError:
                        pass
                    return normalize_extracted_record(data, answer_fields)
                except json.JSONDecodeError as parse_err:
                    print(f"    [DEBUG] Kimi response parse error: {parse_err}")
                    print(f"    [DEBUG] Raw response: {raw[:500]}...")
                    raise RuntimeError(f"Unparseable Kimi response for page {page_num}") from parse_err

            except Exception as e:
                print(f"    Kimi API error (attempt {attempt}/{MAX_RETRIES}): {e}")
                last_error = e

            if attempt < MAX_RETRIES:
                print(f"    Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2

        return _failed_record(last_error, answer_fields)
