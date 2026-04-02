"""Page cells, margin bands, and coordinate helpers (PyMuPDF)."""

from __future__ import annotations

import re

import fitz

from pipeline.models import BBox, ExamImage, Question, flatten_questions

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


def cell_for_point(page: fitz.Page, x: float, y: float) -> fitz.Rect:
    """Mini-page / column rect containing ``(x, y)`` (for clip bounds)."""
    pt = fitz.Point(x, y)
    for cell in page_layout_cells(page):
        if pt in cell:
            return cell
    return page.rect


def expand_bbox_to_subpage_width(doc: fitz.Document, bbox: BBox) -> BBox:
    """Stretch *bbox* horizontally to the enclosing layout cell (2×2 mini-pages, columns, etc.)."""
    pi = int(bbox.page) - 1
    if pi < 0 or pi >= len(doc):
        return bbox
    page = doc[pi]
    cx = (bbox.x0 + bbox.x1) * 0.5
    cy = (bbox.y0 + bbox.y1) * 0.5
    cell = cell_for_point(page, cx, cy)
    return BBox(float(cell.x0), bbox.y0, float(cell.x1), bbox.y1, bbox.page)


def _first_leaf(q: Question) -> Question:
    while q.subquestions:
        q = q.subquestions[0]
    return q


def _last_leaf(q: Question) -> Question:
    while q.subquestions:
        q = q.subquestions[-1]
    return q


def _snap_auxiliary_boxes_to_cell_edge(
    doc: fitz.Document, cfg: ParserConfig, node: Question
) -> None:
    """Stretch writing / answer / image boxes to cell top or bottom when they sit on that edge."""
    pi = int(node.bbox.page) - 1
    if pi < 0 or pi >= len(doc):
        return
    page = doc[pi]
    cx = (node.bbox.x0 + node.bbox.x1) * 0.5
    cy = (node.bbox.y0 + node.bbox.y1) * 0.5
    cell = cell_for_point(page, cx, cy)
    mt, _mb = cell_margin_band(cell, cfg)
    tt = cfg.subpage_edge_snap_tol_top_pt
    tb = cfg.subpage_edge_snap_tol_bottom_pt

    def touch_top(y0: float) -> bool:
        return y0 <= mt + tt

    def touch_bottom(y1: float) -> bool:
        return y1 >= cell.y1 - tb

    for wa in node.writing_areas:
        b = wa.bbox
        y0, y1 = b.y0, b.y1
        if touch_top(y0):
            y0 = float(cell.y0)
        if touch_bottom(y1):
            y1 = float(cell.y1)
        wa.bbox = BBox(b.x0, y0, b.x1, y1, b.page)

    for im in node.images:
        b = im.bbox
        y0, y1 = b.y0, b.y1
        if touch_top(y0):
            y0 = float(cell.y0)
        if touch_bottom(y1):
            y1 = float(cell.y1)
        im.bbox = BBox(b.x0, y0, b.x1, y1, b.page)


def apply_subpage_vertical_snaps(
    doc: fitz.Document,
    cfg: ParserConfig,
    q: Question,
    segment_cell: fitz.Rect,
    snap_top: bool,
    snap_bottom: bool,
) -> None:
    """Snap first exercise in a cell to *cell.y0*, last to *cell.y1*; snap edge-touching aux boxes.

    Between exercises, segment bounds still meet at the next question's first line; *snap_top*
    only applies to the first question in each layout cell.
    """
    cell = segment_cell

    if q.subquestions:
        if snap_top:
            fl = _first_leaf(q)
            fl.bbox = BBox(fl.bbox.x0, float(cell.y0), fl.bbox.x1, fl.bbox.y1, fl.bbox.page)
        if snap_bottom:
            ll = _last_leaf(q)
            ll.bbox = BBox(ll.bbox.x0, ll.bbox.y0, ll.bbox.x1, float(cell.y1), ll.bbox.page)
    else:
        if snap_top:
            q.bbox = BBox(q.bbox.x0, float(cell.y0), q.bbox.x1, q.bbox.y1, q.bbox.page)
        if snap_bottom:
            q.bbox = BBox(q.bbox.x0, q.bbox.y0, q.bbox.x1, float(cell.y1), q.bbox.page)

    for node in flatten_questions([q]):
        _snap_auxiliary_boxes_to_cell_edge(doc, cfg, node)
