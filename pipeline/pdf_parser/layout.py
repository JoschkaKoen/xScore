"""Page cells, margin bands, and coordinate helpers (PyMuPDF)."""

from __future__ import annotations

import re

import fitz

from pipeline.pdf_parser.config import (
    A3_LANDSCAPE_MIN_H,
    A3_LANDSCAPE_MIN_W,
    FOURUP_PORTRAIT_MIN_H,
    FOURUP_PORTRAIT_MIN_W,
    NOMINAL_A4_H,
    NOMINAL_A4_W,
    ParserConfig,
)


def _cells_four_quadrants(r: fitz.Rect) -> list[fitz.Rect]:
    mx = (r.x0 + r.x1) * 0.5
    my = (r.y0 + r.y1) * 0.5
    return [
        fitz.Rect(r.x0, r.y0, mx, my),
        fitz.Rect(mx, r.y0, r.x1, my),
        fitz.Rect(r.x0, my, mx, r.y1),
        fitz.Rect(mx, my, r.x1, r.y1),
    ]


def _page_looks_two_column_portrait(page: fitz.Page) -> bool:
    r = page.rect
    if r.height < r.width:
        return False
    split = r.x0 + r.width * 0.42
    n = 0
    pat = re.compile(r"^(\d{1,2})(\s+|\()")
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not line["spans"]:
                continue
            x0 = line["bbox"][0]
            if x0 < split:
                continue
            t = line["spans"][0]["text"].strip()
            if pat.match(t):
                n += 1
                if n >= 2:
                    return True
    return False


def _line_looks_like_question_anchor(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if re.match(r"^\d{1,2}$", t):
        return True
    if re.match(r"^(\d{1,2})\s*\([a-zA-Z]\)", t):
        return True
    if re.match(r"^(\d{1,2})\s", t):
        return True
    return False


def _page_has_bottom_row_question_anchors(page: fitz.Page) -> bool:
    r = page.rect
    my = (r.y0 + r.y1) * 0.5
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not line["spans"]:
                continue
            y0 = line["bbox"][1]
            if y0 <= my:
                continue
            t = line["spans"][0]["text"].strip()
            if _line_looks_like_question_anchor(t):
                return True
    return False


def page_layout_cells(page: fitz.Page) -> list[fitz.Rect]:
    """Reading-order mini-pages (A4 / A3 / landscape heuristics)."""
    r = page.rect
    portrait = r.height >= r.width

    if portrait:
        if r.width >= FOURUP_PORTRAIT_MIN_W and r.height >= FOURUP_PORTRAIT_MIN_H:
            return _cells_four_quadrants(r)
        if _page_looks_two_column_portrait(page):
            if _page_has_bottom_row_question_anchors(page):
                return _cells_four_quadrants(r)
            mx = (r.x0 + r.x1) * 0.5
            return [
                fitz.Rect(r.x0, r.y0, mx, r.y1),
                fitz.Rect(mx, r.y0, r.x1, r.y1),
            ]
        return [r]

    if r.width >= A3_LANDSCAPE_MIN_W and r.height >= A3_LANDSCAPE_MIN_H:
        return _cells_four_quadrants(r)

    mx = (r.x0 + r.x1) * 0.5
    return [
        fitz.Rect(r.x0, r.y0, mx, r.y1),
        fitz.Rect(mx, r.y0, r.x1, r.y1),
    ]


def cell_scales(cell: fitz.Rect) -> tuple[float, float]:
    return cell.width / NOMINAL_A4_W, cell.height / NOMINAL_A4_H


def cell_margin_band(cell: fitz.Rect, cfg: ParserConfig) -> tuple[float, float]:
    _sx, sy = cell_scales(cell)
    _ = _sx
    top = cell.y0 + cfg.anchor_margin_top * sy
    bottom = cell.y0 + cfg.margin_bottom * sy
    bottom = min(bottom, cell.y1 - 1.0)
    return top, bottom


def bbox_intersects_cell(bbox: tuple[float, float, float, float], cell: fitz.Rect) -> bool:
    br = fitz.Rect(bbox)
    if br.is_empty:
        return False
    return br.intersects(cell)
