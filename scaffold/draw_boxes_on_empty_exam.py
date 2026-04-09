"""Draw color-coded scaffold bounding boxes on a copy of the vector exam PDF.

Each question's ``bbox`` and figure ``images`` get a distinct stroke color derived
from golden-ratio–spaced HSV hues (S=0.82, V=0.92) so overlapping exercises are easy
to tell apart at a glance.

``equation_blank_bboxes`` are drawn in teal on top of the exercise outlines so answer
slots are visually distinct.

``writing_areas`` and ``answer_field_bbox`` are not drawn (the former are always empty;
the latter has been removed from the model).
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import fitz

from shared.models import BBox, Question, flatten_questions

# Golden-ratio increment for hue stepping (φ⁻¹ ≈ 0.6180339887).
_PHI_INV = 0.6180339887
_HUE_S = 0.82
_HUE_V = 0.92

# Teal for equation-blank answer slots.
_TEAL = (0.0, 0.52, 0.55)
_YELLOW: tuple[float, float, float] = (1.0, 0.9, 0.0)


def _hsv_color(index: int) -> tuple[float, float, float]:
    h = (index * _PHI_INV) % 1.0
    return colorsys.hsv_to_rgb(h, _HUE_S, _HUE_V)


def _rects_for_question_node(
    q: Question, color_index: int
) -> list[tuple[int, fitz.Rect, tuple[float, float, float], bool]]:
    """Boxes for this node only; returns ``(page_1, rect, color, is_equation_blank)``."""
    out: list[tuple[int, fitz.Rect, tuple[float, float, float], bool]] = []
    color = _hsv_color(color_index)

    def add(bb: BBox | None, is_eq: bool = False) -> None:
        if bb is None:
            return
        if bb.x1 <= bb.x0 or bb.y1 <= bb.y0:
            return
        c = _TEAL if is_eq else color
        out.append((bb.page, fitz.Rect(bb.x0, bb.y0, bb.x1, bb.y1), c, is_eq))

    add(q.bbox)
    for im in q.images:
        add(im.bbox)
    for eb in q.equation_blank_bboxes:
        add(eb, is_eq=True)
    return out


def write_scaffold_boxes_pdf(
    exam_pdf: Path,
    questions: list[Question],
    output_path: Path | None = None,
    *,
    line_width: float = 0.9,
) -> tuple[Path, int, int]:
    """Copy *exam_pdf* with color-coded outlines for each scaffold region.

    Exercise ``bbox`` and figure ``images`` use distinct golden-ratio HSV colors per
    question (traversal order).  ``equation_blank_bboxes`` are drawn in teal on top.

    Returns ``(output_path, rectangle_count, page_count)``.
    """
    exam_pdf = exam_pdf.resolve()
    if output_path is None:
        output_path = exam_pdf.with_name(f"{exam_pdf.stem}_raw_exam_bboxes.pdf")
    else:
        output_path = output_path.resolve()

    # Collect all rects in traversal order; equation blanks go last so they render on top.
    all_nodes = flatten_questions(questions)
    exercise_rects: list[tuple[int, fitz.Rect, tuple[float, float, float]]] = []
    eq_blank_rects: list[tuple[int, fitz.Rect, tuple[float, float, float]]] = []

    for color_idx, node in enumerate(all_nodes):
        for page_1, rect, color, is_eq in _rects_for_question_node(node, color_idx):
            if is_eq:
                eq_blank_rects.append((page_1, rect, color))
            else:
                exercise_rects.append((page_1, rect, color))

    # Group by page
    by_page: dict[int, list[tuple[fitz.Rect, tuple[float, float, float]]]] = {}
    for page_1, rect, color in exercise_rects + eq_blank_rects:
        by_page.setdefault(page_1, []).append((rect, color))

    doc = fitz.open(exam_pdf)
    try:
        for p1 in sorted(by_page.keys()):
            idx = p1 - 1
            if idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            for rect, color in by_page[p1]:
                page.draw_rect(rect, color=color, width=line_width)
        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()

    total_rects = len(exercise_rects) + len(eq_blank_rects)
    pages_with_marks = len(by_page)
    return output_path, total_rects, pages_with_marks
