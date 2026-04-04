"""Detect which questions each student has attempted (Step F).

One Kimi call per page per student gives a quick overview before
the more detailed per-question grading step.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .kimi_helpers import kimi_image_call, page_to_jpeg_b64, parse_json_safe
from shared.models import ExamScaffold, PageAssignment


def _build_prompt(question_numbers: list[str]) -> str:
    nums = ", ".join(question_numbers)
    return f"""\
The exam contains questions: {nums}

Look at this student's answer sheet page. Which of these questions has the \
student attempted? A question is "attempted" if there is any written answer, \
circled letter, or mark in the answer space — even if incomplete.

Return ONLY a JSON object:
{{"attempted": ["1", "2a", "38"]}}

If none are visible, return:
{{"attempted": []}}
"""


def detect_answered_exercises(
    cleaned_pdf: Path,
    page_map: list[PageAssignment],
    scaffold: ExamScaffold,
    dpi: int = 200,
    client: Any | None = None,
    *,
    pages: list | None = None,
) -> dict[str, list[str]]:
    """Return ``{student_name: [question_numbers_attempted]}`` for every student.

    If *client* is None it is created via ``KimiProvider.create_client()``.
    *pages*: optional pre-rendered page images at *dpi* (skips ``convert_from_path``).
    """
    from pdf2image import convert_from_path

    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()
    if client is None:
        raise RuntimeError("No Kimi client available for answer detection.")

    question_numbers = [q.number for q in scaffold.gradable_questions]
    prompt = _build_prompt(question_numbers)

    from shared.terminal_ui import tool_line

    if pages is None:
        tool_line("detect", f"Rendering pages @ {dpi} DPI …")
        all_pages = convert_from_path(str(cleaned_pdf), dpi=dpi, thread_count=os.cpu_count() or 4)
    else:
        all_pages = pages

    result: dict[str, list[str]] = {}

    for assignment in page_map:
        name = assignment.student_name
        attempted: set[str] = set()

        for page_num in assignment.page_numbers:
            if page_num < 1 or page_num > len(all_pages):
                continue
            page = all_pages[page_num - 1]
            img_b64 = page_to_jpeg_b64(page)
            raw = kimi_image_call(client, img_b64, prompt, max_tokens=256)
            data = parse_json_safe(raw)
            att = data.get("attempted", [])
            if isinstance(att, list):
                page_attempted = [str(x) for x in att if x is not None]
            else:
                page_attempted = []

            attempted.update(page_attempted)
            time.sleep(0.2)

        result[name] = sorted(attempted, key=lambda n: (len(n), n))

    return result
