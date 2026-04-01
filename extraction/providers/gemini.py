"""Google Gemini vision extraction (single call, ensemble, multi-pass voting)."""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import (
    AI_MODEL,
    ENSEMBLE_CALLS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    GEMINI_THINKING,
    GEMINI_THINKING_BUDGET,
    MAX_RETRIES,
    RETRY_BACKOFF_S,
    USE_ENSEMBLE,
)
from extraction.images import normalize_extracted_record


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


class GeminiProvider:
    def extract(
        self,
        client: Any,
        image_bytes: bytes,
        prompt: str,
        schema: type[BaseModel],
        page_num: int,
        answer_fields: list[str],
    ) -> dict:
        if not isinstance(client, genai.Client):
            print("    Error: Gemini model selected but wrong client type")
            return _failed_record("Client type mismatch for Gemini", answer_fields)
        if USE_ENSEMBLE:
            return self._ensemble(client, image_bytes, page_num, prompt, schema, answer_fields, ENSEMBLE_CALLS)
        return self._single(client, image_bytes, page_num, prompt, schema, answer_fields)

    def _single(
        self,
        client: genai.Client,
        image_bytes: bytes,
        page_num: int,
        prompt: str,
        schema: type[BaseModel],
        answer_fields: list[str],
    ) -> dict:
        last_error: Exception | None = None
        backoff = RETRY_BACKOFF_S

        gen_config = types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(
                thinking_budget=GEMINI_THINKING_BUDGET if GEMINI_THINKING else 0
            ),
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=AI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=gen_config,
                )
                try:
                    finish_reason = response.candidates[0].finish_reason
                except (IndexError, AttributeError):
                    finish_reason = "unknown"
                if response.parsed:
                    return normalize_extracted_record(response.parsed.model_dump(), answer_fields)
                raw = response.text or ""
                print(f"\n    [DEBUG] finish_reason={finish_reason}")
                print(f"    [DEBUG] full response ({len(raw)} chars):\n{raw}\n")
                try:
                    return normalize_extracted_record(json.loads(raw), answer_fields)
                except (json.JSONDecodeError, ValueError) as parse_err:
                    raise RuntimeError(
                        f"Unparseable response for page {page_num} (finish_reason={finish_reason})"
                    ) from parse_err

            except Exception as e:
                print(f"    API error (attempt {attempt}/{MAX_RETRIES}): {e}")
                last_error = e

            if attempt < MAX_RETRIES:
                print(f"    Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2

        return _failed_record(last_error, answer_fields)

    def _ensemble(
        self,
        client: genai.Client,
        image_bytes: bytes,
        page_num: int,
        prompt: str,
        schema: type[BaseModel],
        answer_fields: list[str],
        num_calls: int,
    ) -> dict:
        results: list[dict] = []
        for _ in range(num_calls):
            results.append(self._single(client, image_bytes, page_num, prompt, schema, answer_fields))

        if len(results) == 1:
            return results[0]

        final_result = results[0].copy()

        for field in answer_fields:
            votes = [r.get(field, "?") for r in results]
            vote_counts = Counter(votes)
            winner = vote_counts.most_common(1)[0][0]
            final_result[field] = winner

            agreement = vote_counts[winner] / len(votes)
            if agreement == 1.0:
                final_result[f"{field}_confidence"] = "high"
            elif agreement >= 0.5:
                final_result[f"{field}_confidence"] = "medium"
            else:
                final_result[f"{field}_confidence"] = "low"

        names = [
            r.get("student_name", "UNKNOWN")
            for r in results
            if r.get("student_name") not in ("UNKNOWN", "EXTRACTION_ERROR", "?")
        ]
        if names:
            name_counts = Counter(names)
            final_result["student_name"] = name_counts.most_common(1)[0][0]

        confidences = [r.get("confidence", "low") for r in results]
        conf_counts = Counter(confidences)
        final_result["confidence"] = conf_counts.most_common(1)[0][0]

        return final_result
