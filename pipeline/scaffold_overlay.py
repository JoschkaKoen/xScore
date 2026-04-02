"""Draw red rectangles for scaffold bounding boxes on a copy of the vector exam PDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from pipeline.models import BBox, Question, flatten_questions


def _rects_for_question_node(q: Question) -> list[tuple[int, fitz.Rect]]:
    """Boxes attached to this node only (``flatten_questions`` supplies every node)."""
    out: list[tuple[int, fitz.Rect]] = []

    def add(bb: BBox | None) -> None:
        if bb is None:
            return
        if bb.x1 <= bb.x0 or bb.y1 <= bb.y0:
            return
        out.append((bb.page, fitz.Rect(bb.x0, bb.y0, bb.x1, bb.y1)))

    add(q.bbox)
    add(q.answer_field_bbox)
    for w in q.writing_areas:
        add(w.bbox)
    for im in q.images:
        add(im.bbox)
    return out


def write_scaffold_boxes_pdf(
    exam_pdf: Path,
    questions: list[Question],
    output_path: Path | None = None,
    *,
    line_width: float = 0.9,
) -> tuple[Path, int, int]:
    """Copy *exam_pdf* with red box outlines for each scaffold region.

    Draws ``bbox``, ``answer_field_bbox``, ``writing_areas``, and exam ``images`` only
    (not ``answer_images``, which use answer-key coordinates).

    Returns ``(output_path, rectangle_count, page_count)``.
    """
    exam_pdf = exam_pdf.resolve()
    if output_path is None:
        output_path = exam_pdf.with_name(f"{exam_pdf.stem}_scaffold_boxes.pdf")
    else:
        output_path = output_path.resolve()

    rects: list[tuple[int, fitz.Rect]] = []
    for q in flatten_questions(questions):
        rects.extend(_rects_for_question_node(q))

    by_page: dict[int, list[fitz.Rect]] = {}
    for page_1, r in rects:
        by_page.setdefault(page_1, []).append(r)

    doc = fitz.open(exam_pdf)
    try:
        for p1 in sorted(by_page.keys()):
            idx = p1 - 1
            if idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            for r in by_page[p1]:
                page.draw_rect(r, color=(1, 0, 0), width=line_width)
        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()

    return output_path, len(rects), len(by_page)
