"""Moonshot Kimi (OpenAI-compatible) vision extraction."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from config import AI_MODEL, KIMI_MAX_TOKENS, KIMI_THINKING, MAX_RETRIES, RETRY_BACKOFF_S
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


def _pipeline_verbose() -> bool:
    try:
        from shared.terminal_ui import pipeline_verbose

        return pipeline_verbose()
    except Exception:
        return False


def _filter_schema_fields(data: dict, schema: type[BaseModel]) -> dict:
    """Remove extra fields not defined in the schema.
    
    Kimi sometimes adds extra fields like 'notes' or 'overall_confidence'
    that aren't in our schema. This filters them out.
    """
    allowed_fields = set(schema.model_fields.keys())
    return {k: v for k, v in data.items() if k in allowed_fields}


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from text, handling truncation and extra content.

    Tries to find complete JSON object even if response was truncated.
    """
    # Marking-side equivalent: marking.kimi_helpers.parse_json_safe (returns {} on failure).
    text = text.strip()
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object boundaries
    try:
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        pass
    
    # Try to fix truncated strings by closing open quotes
    try:
        # Find unclosed strings and close them
        fixed = text
        # Count quotes - if odd, we have an unclosed string
        if fixed.count('"') % 2 == 1:
            # Add closing quote and brace
            fixed = fixed.rstrip() + '"}'
        # Ensure we end with a closing brace
        if not fixed.rstrip().endswith('}'):
            fixed = fixed.rstrip() + '}'
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


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
        def _warn(msg: str) -> None:
            try:
                from shared.terminal_ui import warn_line
                warn_line(msg)
            except Exception:
                print(msg)

        def _note(msg: str) -> None:
            if not _pipeline_verbose():
                return
            try:
                from shared.terminal_ui import note_line

                note_line(msg)
            except Exception:
                pass

        if not KIMI_AVAILABLE:
            _warn("OpenAI package not installed. Run: pip install openai")
            return None
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            _warn("KIMI_API_KEY not set. Kimi will not be available.")
            return None

        base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        _note("Kimi API client configured.")
        
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
            try:
                from shared.terminal_ui import err_line

                err_line("Kimi model selected but wrong client type")
            except Exception:
                print("Error: Kimi model selected but wrong client type", file=sys.stderr)
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
                response = client.chat.completions.create(**kwargs)

                raw = response.choices[0].message.content or ""
                try:
                    data = json.loads(raw)
                    # Filter out extra fields Kimi might add (notes, overall_confidence, etc.)
                    data = _filter_schema_fields(data, schema)
                    try:
                        schema.model_validate(data)
                    except ValidationError:
                        pass
                    return normalize_extracted_record(data, answer_fields)
                except json.JSONDecodeError as parse_err:
                    # Try to extract partial JSON
                    partial_data = _extract_json_from_text(raw)
                    if partial_data is not None:
                        if _pipeline_verbose():
                            try:
                                from shared.terminal_ui import info_line

                                info_line(
                                    f"Recovered partial JSON for page {page_num}"
                                )
                            except Exception:
                                pass
                        # Also filter extra fields from partial data
                        partial_data = _filter_schema_fields(partial_data, schema)
                        try:
                            schema.model_validate(partial_data)
                        except ValidationError:
                            pass
                        return normalize_extracted_record(partial_data, answer_fields)

                    if _pipeline_verbose():
                        try:
                            from shared.terminal_ui import info_line

                            info_line(f"Kimi parse error: {parse_err}")
                            info_line(f"Raw (500 chars): {raw[:500]}…")
                        except Exception:
                            pass
                    raise RuntimeError(f"Unparseable Kimi response for page {page_num}") from parse_err

            except Exception as e:
                try:
                    from shared.terminal_ui import warn_line

                    warn_line(
                        f"Kimi API error (attempt {attempt}/{MAX_RETRIES}): {e}"
                    )
                except Exception:
                    print(f"Kimi API error (attempt {attempt}/{MAX_RETRIES}): {e}")
                last_error = e

            if attempt < MAX_RETRIES:
                try:
                    from shared.terminal_ui import info_line

                    info_line(f"Retrying in {backoff}s…")
                except Exception:
                    print(f"Retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2

        return _failed_record(last_error, answer_fields)
