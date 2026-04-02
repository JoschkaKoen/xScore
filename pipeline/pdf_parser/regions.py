"""Question anchors, vertical segments, horizontal clips, footer trimming."""

from __future__ import annotations

import re
from collections import defaultdict

import fitz

from pipeline.pdf_parser.config import (
    DEFAULT_PARSER_CONFIG,
    DISCLAIMER_TRIGGERS,
    FOOTER_MARGIN_PT,
    MIN_TRIM_GAP_PT,
    ParserConfig,
    SEPARATOR_MIN_WIDTH_PT,
)
from pipeline.pdf_parser.layout import (
    bbox_intersects_cell,
    cell_margin_band,
    cell_scales,
    page_layout_cells,
)


def get_tight_y_end(
    page: fitz.Page,
    y_start: float,
    y_end: float,
    horiz_band: fitz.Rect | None = None,
) -> float:
    disclaimer_y = y_end
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        if horiz_band is not None and not bbox_intersects_cell(block["bbox"], horiz_band):
            continue
        by0 = block["bbox"][1]
        if by0 < y_start or by0 > y_end:
            continue
        flat = " ".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).lower()
        if any(pat in flat for pat in DISCLAIMER_TRIGGERS):
            disclaimer_y = min(disclaimer_y, by0)

    footer_start_y = disclaimer_y
    if disclaimer_y < y_end:
        for drawing in page.get_drawings():
            r = drawing["rect"]
            if horiz_band is not None and not r.intersects(horiz_band):
                continue
            if r.width < SEPARATOR_MIN_WIDTH_PT:
                continue
            if disclaimer_y - FOOTER_MARGIN_PT <= r.y0 < disclaimer_y:
                footer_start_y = min(footer_start_y, r.y0)

    effective_end = min(y_end, footer_start_y - 2.0)
    last_y = y_start

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        if horiz_band is not None and not bbox_intersects_cell(block["bbox"], horiz_band):
            continue
        by0 = block["bbox"][1]
        if by0 < y_start - 5 or by0 > effective_end:
            continue
        lines = block["lines"]
        if not lines:
            continue
        last_line_y1 = 0.0
        for line in reversed(lines):
            if line["bbox"][1] > effective_end:
                continue
            if any(s["text"].strip() for s in line["spans"]):
                last_line_y1 = min(line["bbox"][3], effective_end)
                break
        if not last_line_y1:
            continue
        last_y = max(last_y, last_line_y1)

    for drawing in page.get_drawings():
        r = drawing["rect"]
        if horiz_band is not None and not r.intersects(horiz_band):
            continue
        if r.y0 < y_start - 5 or r.y0 > effective_end:
            continue
        if r.width < 10:
            continue
        last_y = max(last_y, min(r.y1, effective_end))

    if last_y <= y_start:
        if footer_start_y < y_end - MIN_TRIM_GAP_PT:
            return max(y_start, effective_end)
        return y_end

    tight = min(last_y + 4.0, effective_end)
    if tight < y_end - MIN_TRIM_GAP_PT:
        return tight
    return y_end


def margin_question_raw_number(
    text: str,
    y0: float,
    margin_top: float,
    inline_y_cap: float,
) -> int | None:
    text = text.strip()
    if not text:
        return None
    if re.match(r"^\d{1,2}$", text):
        return int(text)
    m_par = re.match(r"^(\d{1,2})\s*\([a-zA-Z]\)", text)
    if m_par:
        return int(m_par.group(1))
    if y0 <= inline_y_cap:
        m_in = re.match(r"^(\d{1,2})\s", text)
        if m_in:
            return int(m_in.group(1))
    return None


def format_main_question_id(raw: int, occ: int) -> str:
    """First time we see *raw* in the PDF → ``\"9\"``; each further occurrence → ``\"9_2\"``, …"""
    base = str(raw)
    if occ == 1:
        return base
    return f"{base}_{occ}"


def find_question_positions(
    doc: fitz.Document, cfg: ParserConfig = DEFAULT_PARSER_CONFIG
) -> list[tuple[str, int, float, fitz.Rect, int, float]]:
    """Return ``(question_id, page_idx, y0, cell, printed_raw, number_span_x1)`` in layout order.

    *question_id* is the main printed number as a string (``\"38\"``, ``\"38_2\"``, …). Occurrence
    is counted **in document reading order** (all cells/pages), so two ``39`` blocks in different
    columns become ``39`` and ``39_2``. Sub-parts are assigned later as ``39a``, ``39ai``, etc.
    """
    positions: list[tuple[str, int, float, fitz.Rect, int, float]] = []
    raw_occurrence: dict[int, int] = defaultdict(int)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        cells = page_layout_cells(page)
        for cell in cells:
            sx, sy = cell_scales(cell)
            margin_top, margin_bottom = cell_margin_band(cell, cfg)
            qx_max = cell.x0 + cfg.question_x_max * sx
            inline_y_cap = cell.y1 - 2.0

            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    if not line["spans"]:
                        continue
                    if not bbox_intersects_cell(line["bbox"], cell):
                        continue
                    first_span = line["spans"][0]
                    x0 = line["bbox"][0]
                    y0 = line["bbox"][1]
                    if y0 < margin_top or y0 > margin_bottom:
                        continue
                    if x0 > qx_max:
                        continue
                    text = first_span["text"].strip()
                    font_size = first_span["size"]
                    if font_size < cfg.font_size_min or font_size > cfg.font_size_max:
                        continue
                    raw = margin_question_raw_number(text, y0, margin_top, inline_y_cap)
                    if raw is None or not (1 <= raw <= 99):
                        continue
                    raw_occurrence[raw] += 1
                    occ = raw_occurrence[raw]
                    qid = format_main_question_id(raw, occ)
                    num_span_x1 = float(first_span["bbox"][2])
                    positions.append((qid, page_idx, y0, cell, raw, num_span_x1))

    positions.sort(key=lambda x: (x[1], x[3].y0, x[3].x0, x[2]))
    return positions


def iter_region_segments(
    doc: fitz.Document,
    positions: list[tuple[str, int, float, fitz.Rect, int, float]],
    cfg: ParserConfig = DEFAULT_PARSER_CONFIG,
) -> list[tuple[str, int, float, float, fitz.Rect, int, float]]:
    if not positions:
        return []

    def cell_key(t: tuple[str, int, float, fitz.Rect, int, float]) -> tuple[int, float, float]:
        _q, p, _y, c, _raw, _nx = t
        return (p, round(c.y0, 2), round(c.x0, 2))

    groups: dict[tuple[int, float, float], list[tuple[str, int, float, fitz.Rect, int, float]]] = defaultdict(list)
    for t in positions:
        groups[cell_key(t)].append(t)

    ordered_keys = sorted(groups.keys())
    results: list[tuple[str, int, float, float, fitz.Rect, int, float]] = []

    for ck in ordered_keys:
        g = sorted(groups[ck], key=lambda t: t[2])
        cell = g[0][3]
        margin_top, margin_bottom = cell_margin_band(cell, cfg)
        _sx, sy = cell_scales(cell)
        _ = _sx
        pad = cfg.padding_above * sy

        for pos_idx, (qid, q_page, q_y, q_cell, printed_raw, num_x1) in enumerate(g):
            assert q_cell == cell
            if pos_idx + 1 < len(g):
                next_q = g[pos_idx + 1]
                next_page, next_y = next_q[1], next_q[2]
            else:
                next_page = q_page
                next_y = margin_bottom

            pad_above = min(pad, cfg.text_clip_pad_above_pt * sy)
            start_y = max(q_y - pad_above, margin_top)

            end_y = min(next_y - 2.0, margin_bottom)
            end_y = get_tight_y_end(doc[q_page], start_y, end_y, horiz_band=cell)
            results.append((qid, q_page, start_y, end_y, cell, printed_raw, num_x1))

    return results


def clip_horizontal_bounds(
    doc: fitz.Document,
    page_idx: int,
    cfg: ParserConfig,
    cell: fitz.Rect,
) -> tuple[float, float]:
    page_w = doc[page_idx].rect.width
    sx, _sy = cell_scales(cell)
    _ = _sy
    if cell.x0 <= page_w * 0.33:
        left = cell.x0 + cfg.strip_crop_left * sx
    else:
        inner = max(2.0, min(8.0 * sx, cfg.strip_crop_left * sx))
        left = cell.x0 + inner
    right = cell.x1 - cfg.strip_crop_right * sx
    return left, right


def clip_for_segment(
    doc: fitz.Document,
    page_idx: int,
    y0: float,
    y1: float,
    cfg: ParserConfig,
    cell: fitz.Rect,
) -> fitz.Rect:
    left, right = clip_horizontal_bounds(doc, page_idx, cfg, cell)
    return fitz.Rect(left, y0, right, y1)


def clip_for_text_segment(
    doc: fitz.Document,
    page_idx: int,
    y0: float,
    y1: float,
    cfg: ParserConfig,
    cell: fitz.Rect,
    number_span_x1: float | None = None,
) -> fitz.Rect:
    left, right = clip_horizontal_bounds(doc, page_idx, cfg, cell)
    left_base = left
    if number_span_x1 is not None:
        left = max(left, float(number_span_x1))
    if left >= right - 5.0:
        left = left_base
    return fitz.Rect(left, y0, right, y1)
