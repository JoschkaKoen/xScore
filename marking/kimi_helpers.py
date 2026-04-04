"""Shared Kimi multimodal helpers for marking (JPEG, retries, JSON recovery)."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from config import resolve_pipeline_ai_model_id
from extraction.images import to_jpeg_bytes
from shared.terminal_ui import warn_line

# Default: JSON object mode. Pass ``response_format=None`` to omit (non-JSON prompts).
_USE_DEFAULT_JSON_OBJECT = object()


def page_to_jpeg_b64(image: Any, quality: int = 85) -> str:
    """Encode a PIL image as base64 JPEG (quality matches prior marking modules)."""
    return base64.b64encode(to_jpeg_bytes(image, quality=quality)).decode("utf-8")


def kimi_image_call(
    client: Any,
    image_b64: str,
    prompt: str,
    *,
    max_tokens: int = 128,
    response_format: Any = _USE_DEFAULT_JSON_OBJECT,
) -> str:
    """Kimi vision call with retries. Uses :func:`resolve_pipeline_ai_model_id`."""
    model = resolve_pipeline_ai_model_id()
    is_k2_5 = model.startswith("kimi-k2")
    extra: dict[str, Any] = {}
    if is_k2_5:
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    create_kwargs: dict[str, Any] = dict(
        model=model,
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
        max_tokens=max_tokens,
        **extra,
    )
    if response_format is _USE_DEFAULT_JSON_OBJECT:
        create_kwargs["response_format"] = {"type": "json_object"}
    elif response_format is not None:
        create_kwargs["response_format"] = response_format

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            warn_line(f"API error (attempt {attempt}/3): {exc}")
            if attempt < 3:
                time.sleep(2**attempt)
    return ""


def parse_json_safe(raw: str) -> dict:
    """Parse JSON from model text; slice object bounds; light truncation repair.

    Policy aligned with ``extraction.providers.kimi._extract_json_from_text`` (marking
    returns empty dict on failure; extraction returns None).
    """
    text = raw.strip()
    if not text:
        return {}

    def _as_dict(obj: Any) -> dict:
        return obj if isinstance(obj, dict) else {}

    try:
        return _as_dict(json.loads(text))
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return _as_dict(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass

    try:
        fixed = text
        if fixed.count('"') % 2 == 1:
            fixed = fixed.rstrip() + '"}'
        if not fixed.rstrip().endswith("}"):
            fixed = fixed.rstrip() + "}"
        return _as_dict(json.loads(fixed))
    except json.JSONDecodeError:
        pass

    return {}
