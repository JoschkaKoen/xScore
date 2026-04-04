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

import os
import time
from pathlib import Path

from config import PAGE_API_DELAY_S

from .kimi_helpers import KimiChatClient, kimi_image_call, page_to_jpeg_b64, parse_json_safe
from shared.models import PageAssignment


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


def assign_pages(
    cleaned_pdf: Path,
    students: list[str],
    dpi: int = 200,
    client: KimiChatClient | None = None,
    name_crop_fraction: float = 0.15,
    *,
    verbose: bool = True,
    pages: list | None = None,
) -> list[PageAssignment]:
    """Return a ``PageAssignment`` for every student whose pages were found.

    If *client* is None it is created via ``KimiProvider.create_client()``.
    *verbose*: when False (``xscore.py``), log only sparse progress instead of every page.
    *pages*: optional pre-rendered page images at *dpi* (skips ``convert_from_path``).
    """
    from extraction.ground_truth import fuzzy_match_name
    from pdf2image import convert_from_path

    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()
    if client is None:
        raise RuntimeError("No Kimi client available for page assignment.")

    from shared.terminal_ui import info_line, note_line, tool_line

    if pages is None:
        tool_line("pages", f"Rendering pages @ {dpi} DPI …")
        pages = convert_from_path(str(cleaned_pdf), dpi=dpi, thread_count=os.cpu_count() or 4)
    n_pages = len(pages)
    step = max(1, n_pages // 8) if not verbose and n_pages > 1 else 1

    # For each page: ask Kimi for the student name (or empty string)
    raw_names: list[str] = []
    for i, page in enumerate(pages, 1):
        crop = _crop_top(page, fraction=name_crop_fraction)
        img_b64 = page_to_jpeg_b64(crop)
        raw = kimi_image_call(client, img_b64, _NAME_PROMPT, max_tokens=64)
        data = parse_json_safe(raw)
        name = str(data.get("name", "") or "").strip()
        if verbose or i == 1 or i == n_pages or (i % step == 0):
            info_line(f"Page {i:3d}/{n_pages}: raw name = {name!r}")
        raw_names.append(name)
        time.sleep(PAGE_API_DELAY_S)
    if not verbose and n_pages > 1:
        info_line(f"Name OCR: {n_pages} pages (sample every {step}; verbose=all lines)")

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
