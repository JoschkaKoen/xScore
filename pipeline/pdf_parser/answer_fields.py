"""Infer equation-answer blank bounding boxes for written questions.

For each text line that matches the Cambridge-style pattern
``label = ……………… [n]``  (a dotted or spaced blank before a mark bracket)
one ``BBox`` is produced covering the blank area.  These are stored on the
``Question`` as ``equation_blank_bboxes`` and drawn in teal on the overlay.

Multiple-choice questions never get equation blanks — the whole exercise
``bbox`` is used for answer detection on those questions.
"""

from __future__ import annotations

import re

import fitz

from pipeline.models import BBox, Question
from pipeline.pdf_parser.config import ParserConfig
from pipeline.pdf_parser.layout import cell_for_point
from pipeline.pdf_parser.regions import clip_horizontal_bounds

# Matches sub-exercise / sub-sub-exercise structural labels at line start.
_RE_STRUCTURAL_NEXT = re.compile(
    r"^\s*(?:\(?[a-z]\)\s*|\([ivxlcdm]+\)\s*|\d{1,2}\s*\(?[a-zA-Z]?\)?\s)",
    re.I,
)

# Detects "label = ……… [n]" answer lines (dots, spaces, middle-dots, dashes as filler).
_RE_EQUATION_BLANK = re.compile(
    r"=\s*[\s.\u00b7\u2022\-_]{3,}\s*\[\s*\d+\s*\]",
)


def _join_line_text(line: dict) -> str:
    return "".join(s["text"] for s in line["spans"]).strip()


def _lines_in_band(
    page: fitz.Page, x0: float, y0: float, x1: float, y1: float
) -> list[tuple[float, float, float, float, str, list[dict]]]:
    """Return ``(lx0, ly0, lx1, ly1, text, spans)`` for every text line in the rect."""
    out = []
    band = fitz.Rect(x0, y0, x1, y1)
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            bb = line["bbox"]
            r = fitz.Rect(bb)
            if not r.intersects(band):
                continue
            ir = r & band
            if ir.is_empty or ir.get_area() < 0.5:
                continue
            t = _join_line_text(line)
            out.append((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                        t, line["spans"]))
    out.sort(key=lambda row: (row[1], row[0]))
    return out


def _char_x_at_pos(spans: list[dict], char_index: int) -> float | None:
    """Interpolate the x-coordinate of the character at *char_index* across spans."""
    pos = 0
    for sp in spans:
        text = sp["text"]
        n = len(text)
        if char_index < pos + n:
            local = char_index - pos
            x0, _, x1, _ = sp["bbox"]
            if n <= 1:
                return float(x0) if local == 0 else float(x1)
            return x0 + (x1 - x0) * local / n
        pos += n
    return None


def infer_equation_blank_bboxes(
    doc: fitz.Document,
    cfg: ParserConfig,
    q: Question,
) -> list[BBox]:
    """One ``BBox`` per ``label = …… [n]`` line inside *q*'s bbox region."""
    pi = q.bbox.page - 1
    if pi < 0 or pi >= len(doc):
        return []
    page = doc[pi]
    cx = (q.bbox.x0 + q.bbox.x1) * 0.5
    cy = (q.bbox.y0 + q.bbox.y1) * 0.5
    cell = cell_for_point(page, cx, cy)
    h0, h1 = clip_horizontal_bounds(doc, pi, cfg, cell)

    lines = _lines_in_band(page, h0, q.bbox.y0, h1, q.bbox.y1)
    results: list[BBox] = []

    for idx, (lx0, ly0, lx1, ly1, text, spans) in enumerate(lines):
        if not _RE_EQUATION_BLANK.search(text):
            continue

        # Horizontal: from first non-space char after '=' to start of '['
        eq_pos = text.find("=")
        if eq_pos < 0:
            continue
        blank_start_char = eq_pos + 1
        while blank_start_char < len(text) and text[blank_start_char] == " ":
            blank_start_char += 1
        bracket_pos = text.rfind("[")
        if bracket_pos <= blank_start_char:
            continue

        bx0 = _char_x_at_pos(spans, blank_start_char) or lx0
        bx1 = _char_x_at_pos(spans, bracket_pos) or lx1
        bx0 = max(bx0, h0)
        bx1 = min(bx1, h1)
        if bx1 <= bx0:
            continue

        # Vertical: top sits cfg.equation_blank_pad_above_pt above the line top.
        raw_y0 = ly0 - cfg.equation_blank_pad_above_pt

        # Bottom: stop before next structural anchor, or any next non-empty line, or extend
        # a bit if near the cell bottom.
        raw_y1: float = q.bbox.y1  # fallback
        below = [(r[1], r[2], r[4]) for r in lines[idx + 1:] if r[4].strip()]
        for ny0, ny1, nt in below:
            if ny0 <= ly1:
                continue
            if _RE_STRUCTURAL_NEXT.match(nt):
                raw_y1 = ny0 - 2.0
                break
            raw_y1 = ny0 - 2.0
            break
        else:
            # Nothing below — extend if near cell bottom
            if (cell.y1 - ly1) <= cfg.equation_blank_subpage_bottom_tol_pt:
                raw_y1 = ly1 + cfg.equation_blank_pad_below_subpage_pt
            else:
                raw_y1 = min(q.bbox.y1, ly1 + cfg.equation_blank_pad_below_subpage_pt)

        # Apply nudges (push top down, push bottom down for a bit extra).
        raw_y0 += cfg.equation_blank_nudge_top_pt
        raw_y1 += cfg.equation_blank_nudge_bottom_pt

        # Clamp to question band.
        final_y0 = max(raw_y0, q.bbox.y0)
        final_y1 = min(raw_y1, q.bbox.y1)
        if final_y1 <= final_y0 + 2.0:
            continue

        results.append(BBox(bx0, final_y0, bx1, final_y1, q.bbox.page))

    return results


def assign_answer_field_bboxes(doc: fitz.Document, cfg: ParserConfig, q: Question) -> None:
    """Assign ``equation_blank_bboxes`` to *q* and all subquestions recursively.

    Multiple-choice leaves get an empty list (the exercise ``bbox`` covers the answer area).
    """
    if q.question_type == "multiple_choice" or q.subquestions:
        q.equation_blank_bboxes = []
    else:
        q.equation_blank_bboxes = infer_equation_blank_bboxes(doc, cfg, q)
    for sq in q.subquestions:
        assign_answer_field_bboxes(doc, cfg, sq)
