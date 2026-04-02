"""Infer a generous answer-writing band under the last stem line (vector PDF text layout).

Only when the question ``text`` contains an ellipsis placeholder (``...`` or Unicode ``…``),
i.e. a printed dotted answer line. Otherwise ``answer_field_bbox`` is left unset.

``multiple_choice`` questions never get an answer band — use the exercise ``bbox``.
"""

from __future__ import annotations

import re

import fitz

from pipeline.models import BBox, Question
from pipeline.pdf_parser.config import ParserConfig
from pipeline.pdf_parser.layout import cell_for_point
from pipeline.pdf_parser.regions import clip_horizontal_bounds

_RE_STRUCTURAL_NEXT = re.compile(
    r"^\s*(?:\(?[a-z]\)\s*|\([ivxlcdm]+\)\s*|\d{1,2}\s*\(?[a-zA-Z]?\)?\s)",
    re.I,
)


def _text_has_answer_placeholder(text: str) -> bool:
    """True if stem text suggests a dotted write-on line (Cambridge-style ``...``)."""
    t = text or ""
    return "..." in t or "\u2026" in t


def _join_line_text(line: dict) -> str:
    return "".join(s["text"] for s in line["spans"]).strip()


def _is_dot_filler(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    return bool(re.fullmatch(r"[\s.\u00b7\u2022\-_]+", t))


def _is_meaningful_sentence(s: str) -> bool:
    """Printed stem / option / caption — not a dotted answer line."""
    if _is_dot_filler(s):
        return False
    if re.search(r"[A-Za-z]{2,}", s):
        return True
    if re.match(r"^\s*[A-Da-d]\b", s) and len(s) <= 12:
        return True
    if re.search(r"\d", s) and len(s.strip()) >= 5:
        return True
    return len(s.strip()) >= 10


def _lines_in_vertical_band(
    page: fitz.Page, x0: float, y0: float, x1: float, y1: float
) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
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
            out.append((float(bb[1]), float(bb[3]), t))
    out.sort(key=lambda row: (row[0], row[1]))
    return out


def infer_answer_field_bbox(
    doc: fitz.Document,
    cfg: ParserConfig,
    q: Question,
) -> BBox | None:
    """Full exercise width × vertical band from below last stem line to above the next printed line."""
    pi = q.bbox.page - 1
    if pi < 0 or pi >= len(doc):
        return None
    page = doc[pi]
    cx = (q.bbox.x0 + q.bbox.x1) * 0.5
    cy = (q.bbox.y0 + q.bbox.y1) * 0.5
    cell = cell_for_point(page, cx, cy)
    h0, h1 = clip_horizontal_bounds(doc, pi, cfg, cell)

    y_lo = q.bbox.y0
    y_hi = q.bbox.y1
    lines = _lines_in_vertical_band(page, h0, y_lo, h1, y_hi)
    if not lines:
        return None

    meaningful = [(y0, y1, txt) for y0, y1, txt in lines if _is_meaningful_sentence(txt)]
    if not meaningful:
        return None

    # Bottom of printed stem / options (PDF y grows downward).
    content_bottom = max(y1 for y0, y1, _txt in meaningful)

    next_top = y_hi
    for y0, y1, txt in lines:
        if y0 <= content_bottom + 0.25:
            continue
        if _is_meaningful_sentence(txt) or bool(_RE_STRUCTURAL_NEXT.match(txt)):
            next_top = y0
            break

    pad = 1.0
    af_y0 = content_bottom + pad
    af_y1 = next_top - pad
    if af_y1 <= af_y0:
        af_y1 = min(y_hi, af_y0 + 28.0)
    af_y0 = max(y_lo, af_y0)
    af_y1 = min(y_hi, af_y1)
    if af_y1 <= af_y0 + 2.0:
        return None
    return BBox(h0, af_y0, h1, af_y1, q.bbox.page)


def assign_answer_field_bboxes(doc: fitz.Document, cfg: ParserConfig, q: Question) -> None:
    if q.question_type == "multiple_choice" or not _text_has_answer_placeholder(q.text):
        q.answer_field_bbox = None
    else:
        q.answer_field_bbox = infer_answer_field_bbox(doc, cfg, q)
    for sq in q.subquestions:
        assign_answer_field_bboxes(doc, cfg, sq)
