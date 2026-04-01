"""Detect which questions each student has attempted (Step F).

One Kimi call per page per student gives a quick overview before
the more detailed per-question grading step.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from pipeline.models import ExamScaffold, PageAssignment


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


def _to_jpeg_b64(page) -> str:
    buf = io.BytesIO()
    page.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_kimi(client: Any, image_b64: str, prompt: str) -> str:
    model = os.getenv("PIPELINE_AI_MODEL") or "kimi-k2.5"
    is_k2_5 = model.startswith("kimi-k2")
    extra: dict = {}
    if is_k2_5:
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }],
                max_tokens=256,
                response_format={"type": "json_object"},
                **extra,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [answer_detection] API error (attempt {attempt}/3): {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return ""


def detect_answered_exercises(
    cleaned_pdf: Path,
    page_map: list[PageAssignment],
    scaffold: ExamScaffold,
    dpi: int = 200,
    client: Any | None = None,
) -> dict[str, list[str]]:
    """Return ``{student_name: [question_numbers_attempted]}`` for every student.

    If *client* is None it is created via ``KimiProvider.create_client()``.
    """
    from pdf2image import convert_from_path

    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()
    if client is None:
        raise RuntimeError("No Kimi client available for answer detection.")

    question_numbers = [q.number for q in scaffold.questions]
    prompt = _build_prompt(question_numbers)

    print(f"[answer_detection] Rendering {cleaned_pdf.name} at {dpi} DPI …")
    all_pages = convert_from_path(str(cleaned_pdf), dpi=dpi, thread_count=os.cpu_count() or 4)

    result: dict[str, list[str]] = {}

    for assignment in page_map:
        name = assignment.student_name
        attempted: set[str] = set()

        for page_num in assignment.page_numbers:
            if page_num < 1 or page_num > len(all_pages):
                continue
            page = all_pages[page_num - 1]
            img_b64 = _to_jpeg_b64(page)
            raw = _call_kimi(client, img_b64, prompt)

            try:
                data = json.loads(raw)
                page_attempted = [str(x) for x in data.get("attempted", [])]
            except (json.JSONDecodeError, TypeError):
                page_attempted = []

            attempted.update(page_attempted)
            time.sleep(0.2)

        result[name] = sorted(attempted, key=lambda n: (len(n), n))

    return result
