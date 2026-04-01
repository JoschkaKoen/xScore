"""Assign PDF pages to students by reading names from the top of each page.

Step E of the pipeline:
1. Render each page of the cleaned scan PDF.
2. Crop the top 15 % (name area only) and send to Kimi.
3. Fuzzy-match the returned name against the student roster.
4. Group consecutive pages: if page N+1 has no recognisable name, it belongs to
   the same student as page N.

Returns a list of ``PageAssignment`` objects.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from pipeline.models import PageAssignment


_NAME_PROMPT = """\
Look at the top of this exam page. What is the student's name written here?

Return ONLY a JSON object:
{"name": "FirstName LastName"}

If no name is visible or the field is blank, return:
{"name": ""}
"""


def _crop_top(page, fraction: float = 0.15):
    """Return the top *fraction* of a PIL image."""
    w, h = page.size
    return page.crop((0, 0, w, int(h * fraction)))


def _to_jpeg_b64(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_kimi(client: Any, image_b64: str) -> str:
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
                        {"type": "text", "text": _NAME_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }],
                max_tokens=64,
                response_format={"type": "json_object"},
                **extra,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [page_assignment] API error (attempt {attempt}/3): {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return ""


def assign_pages(
    cleaned_pdf: Path,
    students: list[str],
    dpi: int = 200,
    client: Any | None = None,
    name_crop_fraction: float = 0.15,
) -> list[PageAssignment]:
    """Return a ``PageAssignment`` for every student whose pages were found.

    If *client* is None it is created via ``KimiProvider.create_client()``.
    """
    from extraction.ground_truth import fuzzy_match_name
    from pdf2image import convert_from_path

    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()
    if client is None:
        raise RuntimeError("No Kimi client available for page assignment.")

    print(f"[page_assignment] Rendering {cleaned_pdf.name} at {dpi} DPI …")
    pages = convert_from_path(str(cleaned_pdf), dpi=dpi, thread_count=os.cpu_count() or 4)

    # For each page: ask Kimi for the student name (or empty string)
    raw_names: list[str] = []
    for i, page in enumerate(pages, 1):
        crop = _crop_top(page, fraction=name_crop_fraction)
        img_b64 = _to_jpeg_b64(crop)
        raw = _call_kimi(client, img_b64)
        try:
            name = json.loads(raw).get("name", "").strip()
        except (json.JSONDecodeError, AttributeError):
            name = ""
        print(f"  Page {i:3d}/{len(pages)}: raw name = {name!r}")
        raw_names.append(name)
        time.sleep(0.2)  # light rate-limit

    # Fuzzy-match each raw name; empty → "UNKNOWN"
    matched: list[str | None] = []
    for raw in raw_names:
        if raw:
            matched.append(fuzzy_match_name(raw, students))
        else:
            matched.append(None)

    # Group consecutive pages: a None name inherits from the previous match
    assignments: dict[str, list[int]] = {}
    current_student: str | None = None

    for page_num, name in enumerate(matched, 1):
        if name is not None:
            current_student = name
        if current_student is None:
            continue
        assignments.setdefault(current_student, []).append(page_num)

    result = [
        PageAssignment(
            student_name=name,
            page_numbers=pages_list,
            confidence="high" if matched[pages_list[0] - 1] is not None else "low",
        )
        for name, pages_list in assignments.items()
    ]
    return result
