"""Translate a natural-language grading prompt into a structured TaskInstruction.

Uses a text-only Kimi call (no image) so this step is fast and cheap.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from pipeline.shared.models import StudentFilter, TaskInstruction


_SYSTEM_PROMPT = """\
You are an exam grading assistant. Convert the user's natural language grading \
instruction into a JSON object with the following structure:

{
  "task_type": "count_marks" | "check_mc" | "check_answers",
  "student_filter": {
    "mode": "all" | "specific" | "first_n",
    "names": ["Name1", "Name2"],
    "n": 5
  },
  "dpi": 400,
  "folder_hint": "short folder name for matching, or null",
  "folder_path": "explicit relative or absolute path to exam folder, or null",
  "skip_clean_scan": false,
  "force_clean_scan": false,
  "rescaffold": false,
  "through_step": null,
  "no_report": false
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

folder_hint vs folder_path:
- folder_hint: basename or short label for fuzzy match (e.g. "Space Physics Unit Test").
- folder_path: use only if the user gives a path (e.g. "exams/physics_mock"); null otherwise.
- Prefer folder_path when both would apply.

skip_clean_scan (true when user wants to reuse existing cleaned scan without rotate/deskew):
- e.g. "skip cleaning", "use existing cleaned scan", "don't reprocess the scan".

force_clean_scan (true when user wants to ignore cleaned_scan cache and clean again):
- e.g. "re-clean the scan", "force new deskew", "ignore cached cleaned pdf".

rescaffold (true when user wants to rebuild scaffold from exam PDFs):
- e.g. "rebuild scaffold", "reparse the exam", "refresh question boxes".

through_step: integer 1–11 or null. Stop after pipeline step N (README table):
  1 parse prompt, 2 find folder, 3 roster, 4 scaffold, 5 clean scan, 6 assign pages,
  7 exercise detection, 8 grade, 9 print results, 10 ground truth, 11 report.
- e.g. "stop after assigning pages" → 6. Usually null.

no_report (true to skip LaTeX/PDF report):
- e.g. "terminal only", "no pdf report", "skip report".

Never set both skip_clean_scan and force_clean_scan to true.

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
            print(f"  [parse_instruction] API error (attempt {attempt}/3): {exc}")
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
    *dpi_override* (CLI ``--dpi``) takes precedence over DPI from the prompt.
    Other CLI flags OR with the same fields from the parsed JSON (CLI wins for
    ``--through-step`` when provided). ``--folder`` overrides ``folder_path`` /
    ``folder_hint`` from the prompt.

    Falls back to a simple keyword heuristic if the Kimi call fails.
    """
    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()

    instruction = _heuristic_fallback(prompt, dpi_override)

    if client is None:
        print("[parse_instruction] Warning: no Kimi client — using heuristic parse only.")
        return instruction

    raw = _call_kimi_text(client, prompt)
    if not raw.strip():
        print("[parse_instruction] Empty AI response — using heuristic parse.")
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
                print("[parse_instruction] Could not parse AI response — using heuristic parse.")
                return instruction
        else:
            print("[parse_instruction] Could not parse AI response — using heuristic parse.")
            return instruction

    sf_raw = data.get("student_filter") or {}
    if not isinstance(sf_raw, dict):
        sf_raw = {}
    raw_names = sf_raw.get("names")
    if not isinstance(raw_names, list):
        raw_names = []
    raw_n = sf_raw.get("n")
    try:
        n_students = int(raw_n) if raw_n is not None and raw_n != "" else 0
    except (TypeError, ValueError):
        n_students = 0
    student_filter = StudentFilter(
        mode=str(sf_raw.get("mode") or "all"),
        names=[str(x) for x in raw_names if x is not None],
        n=n_students,
    )

    raw_dpi = data.get("dpi")
    try:
        parsed_dpi = int(raw_dpi) if raw_dpi is not None and raw_dpi != "" else 400
    except (TypeError, ValueError):
        parsed_dpi = 400
    dpi = dpi_override or parsed_dpi
    raw_hint = data.get("folder_hint")
    folder_hint = str(raw_hint).strip() if raw_hint not in (None, "") else None
    raw_fp = data.get("folder_path")
    folder_path = str(raw_fp).strip() if raw_fp not in (None, "") else None

    skip_clean_scan = bool(data.get("skip_clean_scan", False))
    force_clean_scan = bool(data.get("force_clean_scan", False))
    rescaffold = bool(data.get("rescaffold", False))
    no_report = bool(data.get("no_report", False))

    ts = data.get("through_step")
    through_step: int | None = None
    if ts is not None and ts != "":
        try:
            n = int(ts)
            if 1 <= n <= 11:
                through_step = n
        except (TypeError, ValueError):
            pass

    if skip_clean_scan and force_clean_scan:
        print("[parse_instruction] Both skip_clean_scan and force_clean_scan in AI JSON — forcing both false.")
        skip_clean_scan = False
        force_clean_scan = False

    return TaskInstruction(
        task_type=data.get("task_type", instruction.task_type),
        student_filter=student_filter,
        dpi=dpi,
        folder_hint=folder_hint,
        folder_path=folder_path,
        skip_clean_scan=skip_clean_scan,
        force_clean_scan=force_clean_scan,
        rescaffold=rescaffold,
        through_step=through_step,
        no_report=no_report,
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

    skip_clean = "skip" in p and ("clean" in p or "deskew" in p or "scan" in p)
    force_clean = ("force" in p and "clean" in p) or "re-clean" in p or "reclean" in p.replace(" ", "")
    # avoid double-trigger: simple heuristics
    if skip_clean and force_clean:
        skip_clean = force_clean = False

    return TaskInstruction(
        task_type=task_type,
        student_filter=student_filter,
        dpi=dpi,
        rescaffold="rescaffold" in p or "reparse" in p or "rebuild scaffold" in p,
        skip_clean_scan=skip_clean,
        force_clean_scan=force_clean,
        no_report="no report" in p or "terminal only" in p,
    )
