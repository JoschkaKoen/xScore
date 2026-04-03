"""Translate a natural-language grading prompt into a structured TaskInstruction.

Uses a text-only Kimi call (no image) so this step is fast and cheap.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from pipeline.models import StudentFilter, TaskInstruction


_SYSTEM_PROMPT = """\
You are an exam grading assistant. Convert the user's natural language grading \
instruction into a JSON object with the following structure:

{
  "task_type": "count_marks" | "check_mc" | "check_answers",
  "student_filter": {
    "mode": "all" | "specific" | "first_n",
    "names": ["Name1", "Name2"],   // only when mode = "specific"
    "n": 5                         // only when mode = "first_n"
  },
  "dpi": 400,
  "folder_hint": "folder name or null"
}

Task type rules:
- "count_marks"   → teacher already marked in red; just tally the written scores.
- "check_mc"      → verify only multiple-choice answers against the answer key.
- "check_answers" → verify ALL question types against the answer key.

Student filter rules:
- Default mode is "all".
- "specific" if the prompt names individual students.
- "first_n" if the prompt says "first N students" or similar.

DPI rules:
- Default is 400.
- Use 300 if the prompt says "fast" or "quick".
- Use 600 if the prompt says "high quality" or "accurate".

folder_hint rules:
- Extract a folder name if the prompt mentions one explicitly (e.g. "Space Physics folder").
- Otherwise null.

Return ONLY the JSON object, no explanation.
"""


def _call_kimi_text(client: Any, user_message: str) -> str:
    """Make a text-only Kimi chat call and return the raw response string."""
    from config import KIMI_MAX_TOKENS, KIMI_THINKING  # local import to keep module lightweight

    model = os.getenv("PIPELINE_AI_MODEL") or "kimi-k2.5"
    is_k2_5 = model.startswith("kimi-k2")
    extra: dict = {}
    if is_k2_5:
        thinking_type = "enabled" if KIMI_THINKING else "disabled"
        extra["extra_body"] = {"thinking": {"type": thinking_type}}

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=KIMI_MAX_TOKENS,
        response_format={"type": "json_object"},
        **extra,
    )

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [prompt_parser] API error (attempt {attempt}/3): {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return ""


def parse_prompt(
    prompt: str,
    client: Any | None = None,
    dpi_override: int | None = None,
) -> TaskInstruction:
    """Parse *prompt* into a ``TaskInstruction``.

    If *client* is None it is created automatically using ``KIMI_API_KEY``.
    *dpi_override* (from the CLI --dpi flag) takes precedence over any DPI
    extracted from the prompt.

    Falls back to a simple keyword heuristic if the Kimi call fails.
    """
    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()

    instruction = _heuristic_fallback(prompt, dpi_override)

    if client is None:
        print("[prompt_parser] Warning: no Kimi client — using heuristic parse only.")
        return instruction

    raw = _call_kimi_text(client, prompt)
    if not raw.strip():
        print("[prompt_parser] Empty AI response — using heuristic parse.")
        return instruction

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block in response
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                print("[prompt_parser] Could not parse AI response — using heuristic parse.")
                return instruction
        else:
            print("[prompt_parser] Could not parse AI response — using heuristic parse.")
            return instruction

    sf_raw = data.get("student_filter", {})
    student_filter = StudentFilter(
        mode=sf_raw.get("mode", "all"),
        names=sf_raw.get("names", []),
        n=int(sf_raw.get("n", 0)),
    )

    dpi = dpi_override or int(data.get("dpi", 400))
    folder_hint = data.get("folder_hint") or None

    return TaskInstruction(
        task_type=data.get("task_type", instruction.task_type),
        student_filter=student_filter,
        dpi=dpi,
        folder_hint=folder_hint,
    )


def _heuristic_fallback(prompt: str, dpi_override: int | None) -> TaskInstruction:
    """Simple keyword-based parse used when the AI call fails."""
    p = prompt.lower()

    if "count" in p and "mark" in p:
        task_type = "count_marks"
    elif "multiple choice" in p or " mc " in p or "check mc" in p:
        task_type = "check_mc"
    else:
        task_type = "check_answers"

    student_filter = StudentFilter()
    if "first" in p:
        m = re.search(r"first\s+(\d+)", p)
        if m:
            student_filter = StudentFilter(mode="first_n", n=int(m.group(1)))

    dpi = dpi_override or (300 if ("fast" in p or "quick" in p) else 400)

    return TaskInstruction(task_type=task_type, student_filter=student_filter, dpi=dpi)
