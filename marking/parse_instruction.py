"""Translate a natural-language grading prompt into a structured TaskInstruction.

Uses a text-only Kimi call (no image) so this step is fast and cheap.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .kimi_helpers import parse_json_safe
from shared.models import StudentFilter, TaskInstruction
from shared.terminal_ui import info_line, warn_line


_SYSTEM_PROMPT = """\
Convert the grading instruction to JSON. Return ONLY the JSON, no explanation.

{
  "task_type": "count_marks|check_mc|check_answers",
  "student_filter": {"mode": "all|specific|first_n", "names": [], "n": 0},
  "dpi": 400,
  "folder_hint": null,
  "folder_path": null,
  "skip_clean_scan": false,
  "force_clean_scan": false,
  "rescaffold": false,
  "through_step": null,
  "no_report": false
}

task_type: count_marks=tally red teacher marks; check_mc=MC only; check_answers=all types.
student_filter.mode: all=default; specific=named students; first_n=first N (set n). names=list.
dpi: 400 default; 300 if "fast"/"quick"; 600 if "high quality"/"accurate".
folder_hint: short name for fuzzy folder match. folder_path: only if user gives explicit path; else null.
Prefer folder_path when both apply.
skip_clean_scan: true=reuse cleaned scan ("skip cleaning", "don't reprocess").
force_clean_scan: true=ignore cache, re-clean ("re-clean", "force deskew"). Never both true.
rescaffold: true=rebuild scaffold ("rebuild scaffold", "reparse", "refresh questions").
through_step: 1-11 or null. 1=parse, 2=find folder, 3=roster, 4=scaffold, 5=clean scan,
  6=assign pages, 7=exercises, 8=grade, 9=print results, 10=ground truth, 11=report.
no_report: true=skip PDF ("terminal only", "no report").
"""


def _call_kimi_text(client: Any, user_message: str) -> str:
    """Make a text-only Kimi chat call and return the raw response string."""
    from config import (  # local import to keep module lightweight
        KIMI_THINKING,
        PARSE_PROMPT_MAX_TOKENS,
        resolve_pipeline_ai_model_id,
    )

    model = resolve_pipeline_ai_model_id()
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
        max_tokens=PARSE_PROMPT_MAX_TOKENS,
        response_format={"type": "json_object"},
        **extra,
    )
    # kimi-k2.x rejects non-default temperature (400); other models benefit from 0 for JSON.
    if not model.startswith("kimi-k2"):
        kwargs["temperature"] = 0

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            warn_line(f"Parse prompt API error (attempt {attempt}/3): {exc}")
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
        warn_line("No Kimi client — using heuristic parse only.")
        return instruction

    raw = _call_kimi_text(client, prompt)
    if not raw.strip():
        warn_line("Empty AI response — using heuristic parse.")
        return instruction

    data = parse_json_safe(raw)
    if not data:
        warn_line("Could not parse AI response — using heuristic parse.")
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

    mode_raw = str(sf_raw.get("mode") or "all").strip().lower().replace("-", "_").replace(" ", "_")
    if mode_raw not in ("all", "specific", "first_n"):
        warn_line(f"Unknown student_filter.mode {sf_raw.get('mode')!r} — using 'all'.")
        mode_raw = "all"
    names = [str(x) for x in raw_names if x is not None]

    if mode_raw == "specific" and not names:
        warn_line("student_filter specific had empty names — using 'all'.")
        mode_raw = "all"
    if mode_raw == "first_n" and n_students <= 0:
        warn_line("student_filter first_n had invalid n — using 'all'.")
        mode_raw = "all"
        n_students = 0

    student_filter = StudentFilter(
        mode=mode_raw,
        names=names if mode_raw == "specific" else [],
        n=n_students if mode_raw == "first_n" else 0,
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
        info_line("AI JSON had both skip_clean_scan and force_clean_scan — cleared both.")
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
            k = int(m.group(1))
            if k > 0:
                student_filter = StudentFilter(mode="first_n", n=k)

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
