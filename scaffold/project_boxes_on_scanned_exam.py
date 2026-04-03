"""Project scaffold bounding boxes from 4-up raw-exam PDF coordinates onto
deskewed scan page pixel coordinates.

Background
----------
The scaffold is built from a single-page 4-up PDF (595.3 x 841.9 pt) whose
four sub-pages are arranged as::

    [ sub 1 (TL) | sub 2 (TR) ]
    [ sub 3 (BL) | sub 4 (BR) ]

The deskewed scan is an A3 raster page at 300 DPI (3508 x ~4961 px).  After
deskewing it is split at the vertical midpoint into a **top half** and a
**bottom half**, each containing one landscape A4 sheet with two sub-pages
side by side.

Each sub-page header carries an "IGCSE Physics: sXX YY" label, whose center
coordinates are:
- Known exactly in 4-up PDF space (extracted from the raw vector PDF).
- Detected on the deskewed scan via template matching (stored in the
  ``*_anchors.json`` sidecar from :func:`preprocessing.deskew.deskew_pdf_raster`;
  legacy ``*_reflines.json`` is still readable).

These two pairs of corresponding points define a **similarity transform**
(uniform scale + translation) per half-page.  Rotation is already handled by
the deskew step, so no rotation term is needed.

Transform math (per half)
-------------------------
Given raw anchor pair ``(raw_L, raw_R)`` and scan anchor pair
``(scan_L, scan_R)``::

    scale = (scan_R.x − scan_L.x) / (raw_R.x − raw_L.x)
    tx    = scan_L.x − scale × raw_L.x
    ty    = mean(scan_L.y, scan_R.y) − scale × raw_L.y

    scan_x = scale × raw_x + tx
    scan_y = scale × raw_y + ty

Bbox coordinates that straddle the 4-up midpoint (y = 420.9 pt) use the
``top_transform`` when ``y0 < mid_y``, else the ``bot_transform``.

Usage example
-------------
::

    from pathlib import Path
    from scaffold.project_boxes_on_scanned_exam import (
        extract_raw_igcse_anchors,
        compute_page_transforms,
        project_scaffold_bbox,
    )

    raw_anchors  = extract_raw_igcse_anchors(Path("raw exam 4up.pdf"))
    scan_page    = reflines_data[0]   # one entry from anchors / legacy reflines sidecar
    top_tf, bot_tf = compute_page_transforms(raw_anchors, scan_page["anchors"])

    # Project a Question.bbox (BBox dataclass from shared.models)
    x0, y0, x1, y1 = project_scaffold_bbox(question.bbox, top_tf, bot_tf)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import fitz  # PyMuPDF

from shared.models import BBox, Question, flatten_questions
from scaffold.draw_boxes_on_empty_exam import _TEAL, _hsv_color

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: y-coordinate (pt) that divides the 4-up page into top and bottom halves.
_RAW_MID_Y_PT: float = 420.9

#: Horizontal midpoint (pt) of the 4-up page — divides left from right sub-pages.
_RAW_MID_X_PT: float = 297.6


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class _Point(NamedTuple):
    x: float
    y: float


@dataclass
class SimilarityTransform:
    """Uniform-scale + translation transform from 4-up PDF points to scan pixels.

    Attributes:
        scale: Pixels per PDF point.
        tx:    X translation (px).
        ty:    Y translation (px).
    """
    scale: float
    tx: float
    ty: float

    def project_point(self, x_pt: float, y_pt: float) -> tuple[float, float]:
        """Map one point from 4-up PDF space to half-page pixel space."""
        return self.scale * x_pt + self.tx, self.scale * y_pt + self.ty

    def project_bbox(
        self, x0_pt: float, y0_pt: float, x1_pt: float, y1_pt: float
    ) -> tuple[float, float, float, float]:
        """Map a rectangle from 4-up PDF space to half-page pixel space."""
        x0, y0 = self.project_point(x0_pt, y0_pt)
        x1, y1 = self.project_point(x1_pt, y1_pt)
        return x0, y0, x1, y1

    def __str__(self) -> str:
        return f"scale={self.scale:.4f} px/pt  tx={self.tx:.1f}  ty={self.ty:.1f}"


# ---------------------------------------------------------------------------
# Locate the vector 4-up exam PDF (IGCSE anchor geometry)
# ---------------------------------------------------------------------------

def find_raw_four_up_pdf(folder: Path) -> Path | None:
    """Return a raw exam PDF in *folder* whose name suggests a 4-up imposition.

    Projection uses :func:`extract_raw_igcse_anchors`, which expects one page with
    four quadrant headers. Skips answer keys and scans.
    """
    folder = Path(folder)
    exact = folder / "raw exam 4up.pdf"
    if exact.is_file():
        return exact
    cands = sorted(
        (
            p
            for p in folder.glob("*.pdf")
            if "4up" in p.name.lower()
            and "answer" not in p.name.lower()
            and "scan" not in p.name.lower()
        ),
        key=lambda p: p.name.lower(),
    )
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Extract reference anchors from the raw 4-up PDF
# ---------------------------------------------------------------------------

def extract_raw_igcse_anchors(raw_4up_pdf: Path) -> dict[str, tuple[float, float]]:
    """Return the four top-of-subpage IGCSE header positions from *raw_4up_pdf*.

    The raw 4-up PDF contains "IGCSE Physics: sXX YY" labels at the top of
    each sub-page quadrant.  Some sub-pages also have a *second* scattered
    "IGCSE" line further down — those are ignored by selecting only the
    **topmost** (smallest y) label in each quadrant.

    Args:
        raw_4up_pdf: Path to the single-page 4-up PDF
            (e.g. ``"raw exam 4up.pdf"``).

    Returns:
        Dict with keys ``top_left``, ``top_right``, ``bot_left``,
        ``bot_right``; each value is ``(x_pt, y_pt)`` — center of the
        "IGCSE …" line in PDF point space.

    Raises:
        ValueError: If fewer than 4 distinct IGCSE anchor positions are found.
    """
    raw_4up_pdf = Path(raw_4up_pdf)
    doc = fitz.open(str(raw_4up_pdf))
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height
    mid_x = pw / 2
    mid_y = ph / 2

    # Collect all IGCSE line centers, deduplicated to 1-pt grid
    igcse_centers: list[tuple[float, float, str]] = []  # (cx, cy, text)
    seen: set[tuple[int, int]] = set()
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if "IGCSE" not in text:
                continue
            bb = line["bbox"]
            cx = (bb[0] + bb[2]) / 2
            cy = (bb[1] + bb[3]) / 2
            key = (round(cx), round(cy))
            if key not in seen:
                seen.add(key)
                igcse_centers.append((cx, cy, text))

    doc.close()

    # For each quadrant keep only the topmost (smallest cy) label
    best: dict[str, tuple[float, float] | None] = {
        "top_left": None, "top_right": None,
        "bot_left": None, "bot_right": None,
    }
    for cx, cy, _text in igcse_centers:
        key = (
            ("top" if cy < mid_y else "bot")
            + "_"
            + ("left" if cx < mid_x else "right")
        )
        if best[key] is None or cy < best[key][1]:  # type: ignore[index]
            best[key] = (cx, cy)

    missing = [k for k, v in best.items() if v is None]
    if missing:
        raise ValueError(
            f"[project_boxes_on_scanned_exam] Could not find IGCSE anchors in {raw_4up_pdf.name} "
            f"for quadrant(s): {missing}"
        )

    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Transform computation
# ---------------------------------------------------------------------------

def compute_half_transform(
    raw_left:  tuple[float, float],
    raw_right: tuple[float, float],
    scan_left:  tuple[float, float],
    scan_right: tuple[float, float],
) -> SimilarityTransform:
    """Compute a similarity transform from one pair of corresponding anchors.

    Args:
        raw_left:   ``(x, y)`` of the left IGCSE anchor in 4-up PDF pt.
        raw_right:  ``(x, y)`` of the right IGCSE anchor in 4-up PDF pt.
        scan_left:  ``(x, y)`` of the left IGCSE anchor in scan pixels
                    (half-page coordinates).
        scan_right: ``(x, y)`` of the right IGCSE anchor in scan pixels
                    (half-page coordinates).

    Returns:
        :class:`SimilarityTransform` mapping PDF pt → scan px.
    """
    dx_raw  = raw_right[0]  - raw_left[0]
    dx_scan = scan_right[0] - scan_left[0]
    scale = dx_scan / dx_raw
    tx = scan_left[0] - scale * raw_left[0]
    # Average the two y observations to reduce noise (both should map to same raw y)
    ty = (scan_left[1] + scan_right[1]) / 2.0 - scale * raw_left[1]
    return SimilarityTransform(scale=scale, tx=tx, ty=ty)


def compute_page_transforms(
    raw_anchors: dict[str, tuple[float, float]],
    scan_anchors: dict[str, dict | None],
) -> tuple[SimilarityTransform, SimilarityTransform]:
    """Return ``(top_transform, bot_transform)`` for one scanned page.

    Args:
        raw_anchors:  Output of :func:`extract_raw_igcse_anchors`.
        scan_anchors: The ``"anchors"`` sub-dict from one entry in the
                      anchor sidecar (``*_anchors.json`` or legacy ``*_reflines.json``).
                      Values are dicts with
                      ``"x"``, ``"y"``, ``"score"`` keys (or ``None``).

    Returns:
        ``(top_transform, bot_transform)`` as :class:`SimilarityTransform`.

    Raises:
        ValueError: If any required anchor is missing (``None``) in
            *scan_anchors*.
    """
    for key in ("top_left", "top_right", "bot_left", "bot_right"):
        if scan_anchors.get(key) is None:
            raise ValueError(
                f"[project_boxes_on_scanned_exam] Scan anchor '{key}' is missing "
                f"(template match failed for this page)."
            )

    def _scan(key: str) -> tuple[float, float]:
        a = scan_anchors[key]
        return float(a["x"]), float(a["y"])  # type: ignore[index]

    top_tf = compute_half_transform(
        raw_left=raw_anchors["top_left"],
        raw_right=raw_anchors["top_right"],
        scan_left=_scan("top_left"),
        scan_right=_scan("top_right"),
    )
    bot_tf = compute_half_transform(
        raw_left=raw_anchors["bot_left"],
        raw_right=raw_anchors["bot_right"],
        scan_left=_scan("bot_left"),
        scan_right=_scan("bot_right"),
    )
    return top_tf, bot_tf


# ---------------------------------------------------------------------------
# Bbox projection
# ---------------------------------------------------------------------------

def project_scaffold_bbox(
    bbox,
    top_transform: SimilarityTransform,
    bot_transform: SimilarityTransform,
    mid_y: float = _RAW_MID_Y_PT,
) -> tuple[float, float, float, float]:
    """Project one scaffold bbox from 4-up PDF space to half-page scan pixels.

    The bbox coordinate ``y0`` determines which transform is used:
    - ``y0 < mid_y``  → ``top_transform``; result is in the **top** half-page.
    - ``y0 >= mid_y`` → ``bot_transform``; result is in the **bottom** half-page.

    In both cases the returned coordinates are relative to the **top of that
    half** (y=0 = first row of the top or bottom half-page image).

    Args:
        bbox:          A :class:`shared.models.BBox` (or any object with
                       ``.x0``, ``.y0``, ``.x1``, ``.y1`` attributes), in
                       PDF points on the 4-up page.
        top_transform: Transform for bboxes whose ``y0`` is in the top half.
        bot_transform: Transform for bboxes whose ``y0`` is in the bottom half.
        mid_y:         The y-coordinate (pt) that divides top from bottom half
                       on the 4-up page.  Defaults to 420.9 pt.

    Returns:
        ``(x0_px, y0_px, x1_px, y1_px)`` in half-page pixel coordinates
        (floats; caller should round/clip as needed for cropping).
    """
    tf = top_transform if bbox.y0 < mid_y else bot_transform
    return tf.project_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)


def project_all_scaffold_bboxes(
    questions: list,
    top_transform: SimilarityTransform,
    bot_transform: SimilarityTransform,
    mid_y: float = _RAW_MID_Y_PT,
) -> list[dict]:
    """Project every scaffold question's bbox and return a flat list of records.

    Walks the full question tree (including nested subquestions) and projects
    each bbox.  Useful for validation and debugging.

    Args:
        questions:     Root-level questions list from an :class:`ExamScaffold`.
        top_transform: See :func:`project_scaffold_bbox`.
        bot_transform: See :func:`project_scaffold_bbox`.
        mid_y:         See :func:`project_scaffold_bbox`.

    Returns:
        List of dicts, one per question node, with keys:
        ``number``, ``half`` ("top"/"bot"), ``raw_bbox`` (x0,y0,x1,y1),
        ``scan_bbox`` (x0_px,y0_px,x1_px,y1_px).
    """
    results: list[dict] = []

    def _walk(q) -> None:
        bbox = q.bbox
        half = "top" if bbox.y0 < mid_y else "bot"
        scan = project_scaffold_bbox(bbox, top_transform, bot_transform, mid_y)
        results.append({
            "number":   q.number,
            "half":     half,
            "raw_bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
            "scan_bbox": scan,
        })
        for sub in q.subquestions:
            _walk(sub)

    for q in questions:
        _walk(q)
    return results


# ---------------------------------------------------------------------------
# Draw projected boxes on a deskewed raster scan PDF
# ---------------------------------------------------------------------------

def _half_page_px_to_page_rect(
    x0_px: float,
    y0_px: float,
    x1_px: float,
    y1_px: float,
    half: str,
    mid_px: int,
    px_to_pt: float,
) -> fitz.Rect:
    """Map half-page pixel bbox to full-page PDF coordinates (points, top-left origin)."""
    xa, xb = sorted((x0_px, x1_px))
    ya, yb = sorted((y0_px, y1_px))
    y_off = 0 if half == "top" else float(mid_px)
    return fitz.Rect(
        xa * px_to_pt,
        (ya + y_off) * px_to_pt,
        xb * px_to_pt,
        (yb + y_off) * px_to_pt,
    )


def _projected_items_for_question_node(
    q: Question,
    color_index: int,
    top_tf: SimilarityTransform,
    bot_tf: SimilarityTransform,
    *,
    scaffold_page: int = 1,
    mid_y_pt: float = _RAW_MID_Y_PT,
) -> list[tuple[str, tuple[float, float, float, float], tuple[float, float, float], bool]]:
    """Like ``draw_boxes_on_empty_exam._rects_for_question_node`` but in projected scan space.

    Returns tuples ``(half, (x0,y0,x1,y1)_px, rgb, is_equation_blank)`` in **half-page
    pixel** coordinates (``half`` is ``\"top\"`` or ``\"bot\"`` for y-offset on the page).
    """
    out: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float], bool]] = []
    color = _hsv_color(color_index)

    def add(bb: BBox | None, *, is_eq: bool = False) -> None:
        if bb is None:
            return
        if bb.page != scaffold_page:
            return
        if bb.x1 <= bb.x0 or bb.y1 <= bb.y0:
            return
        half = "top" if bb.y0 < mid_y_pt else "bot"
        quad = project_scaffold_bbox(bb, top_tf, bot_tf, mid_y_pt)
        c = _TEAL if is_eq else color
        out.append((half, quad, c, is_eq))

    add(q.bbox)
    for im in q.images:
        add(im.bbox)
    for eb in q.equation_blank_bboxes:
        add(eb, is_eq=True)
    return out


def overlay_projected_scaffold_on_scan_pdf(
    deskewed_pdf: Path,
    reflines_json: Path,
    raw_4up_pdf: Path,
    questions: list[Question],
    output_pdf: Path,
    *,
    dpi: int = 300,
    line_width: float = 0.9,
    scaffold_page: int = 1,
    mid_y_pt: float = _RAW_MID_Y_PT,
    verbose: bool = True,
) -> Path:
    """Draw projected scaffold regions on a **copy** of the deskewed scan PDF.

    For each raster page, reads that page's IGCSE anchors from *reflines_json*,
    computes top/bottom similarity transforms, projects every question bbox
    (plus images and equation-blank boxes) from 4-up PDF space onto scan
    pixels, then strokes rectangles in PDF point space.  Colours follow the same
    golden-ratio scheme as :func:`scaffold.draw_boxes_on_empty_exam.write_scaffold_boxes_pdf`;
    equation blanks use teal.

    Args:
        deskewed_pdf: Output of :func:`preprocessing.deskew.deskew_pdf_raster`.
        reflines_json: Anchor sidecar (``*_anchors.json`` or legacy ``*_reflines.json``)
            with ``anchors`` per page.
        raw_4up_pdf: Raw exam PDF used to build the scaffold (4-up layout).
        questions: Root-level questions from :class:`shared.models.ExamScaffold`.
        output_pdf: Destination path (must differ from *deskewed_pdf* unless you
            intend to overwrite after loading into memory first — caller's choice).
        dpi: Rasterisation DPI of *deskewed_pdf* (default 300).
        line_width: Stroke width in PDF points.
        scaffold_page: Only draw ``BBox`` objects whose ``page`` equals this (1-based).
        mid_y_pt: 4-up split line for top vs bottom transform (default 420.9).
        verbose: Per-page progress lines; set False for a single summary line (pipeline).

    Returns:
        Path to the written *output_pdf*.
    """
    deskewed_pdf = Path(deskewed_pdf)
    reflines_json = Path(reflines_json)
    raw_4up_pdf = Path(raw_4up_pdf)
    output_pdf = Path(output_pdf)

    use_tmp = output_pdf.resolve() == deskewed_pdf.resolve()
    save_path = output_pdf.with_suffix(".bbox_overlay_tmp.pdf") if use_tmp else output_pdf

    raw_anchors = extract_raw_igcse_anchors(raw_4up_pdf)
    sidecar: list[dict] = json.loads(reflines_json.read_text())
    px_to_pt = 72.0 / dpi

    all_nodes = flatten_questions(questions)

    from shared.terminal_ui import ok_line, tool_line, warn_line

    doc = fitz.open(str(deskewed_pdf))
    try:
        n_doc = len(doc)
        n_side = len(sidecar)
        if n_side != n_doc:
            warn_line(
                f"[bbox_overlay] sidecar has {n_side} pages, PDF has {n_doc} "
                f"— overlaying min({n_side}, {n_doc}) pages"
            )

        total_rects = 0
        n_overlay = min(n_doc, n_side)
        for page_idx in range(n_overlay):
            entry = sidecar[page_idx]
            page = doc[page_idx]
            h_px = int(round(page.rect.height / px_to_pt))
            mid_px = h_px // 2

            top_tf, bot_tf = compute_page_transforms(raw_anchors, entry["anchors"])

            exercise: list[tuple[fitz.Rect, tuple[float, float, float]]] = []
            eq_blank: list[tuple[fitz.Rect, tuple[float, float, float]]] = []

            for color_idx, node in enumerate(all_nodes):
                for half, quad, color, is_eq in _projected_items_for_question_node(
                    node,
                    color_idx,
                    top_tf,
                    bot_tf,
                    scaffold_page=scaffold_page,
                    mid_y_pt=mid_y_pt,
                ):
                    x0, y0, x1, y1 = quad
                    r = _half_page_px_to_page_rect(
                        x0, y0, x1, y1, half, mid_px, px_to_pt
                    )
                    r = r.intersect(page.rect)
                    if r.is_empty:
                        continue
                    if is_eq:
                        eq_blank.append((r, color))
                    else:
                        exercise.append((r, color))

            for r, color in exercise + eq_blank:
                page.draw_rect(r, color=color, width=line_width)

            n_drawn = len(exercise) + len(eq_blank)
            total_rects += n_drawn
            if verbose:
                tool_line(
                    "overlay",
                    f"page {page_idx + 1}/{n_doc} · {n_drawn} rects "
                    f"(exercise={len(exercise)} eq_blank={len(eq_blank)})",
                )

        doc.save(str(save_path), garbage=4, deflate=True)
    finally:
        doc.close()

    if use_tmp:
        save_path.replace(output_pdf)

    if verbose:
        tool_line("overlay", f"Saved → {output_pdf}")
    else:
        ok_line(
            f"Scaffold overlay: {total_rects} boxes / {n_overlay} pages → {output_pdf.name}",
        )
    return output_pdf


# ---------------------------------------------------------------------------
# CLI / quick validation helper
# ---------------------------------------------------------------------------

def _print_page_transforms(
    raw_4up_pdf: Path,
    reflines_json: Path,
    page_number: int = 1,
) -> None:
    """Print transforms and projected bboxes for *page_number* (1-based)."""
    raw_anchors = extract_raw_igcse_anchors(raw_4up_pdf)
    data = json.loads(Path(reflines_json).read_text())

    entry = next((e for e in data if e["page"] == page_number), None)
    if entry is None:
        print(f"Page {page_number} not found in {reflines_json.name}")
        return

    top_tf, bot_tf = compute_page_transforms(raw_anchors, entry["anchors"])
    print(f"Page {page_number}:")
    print(f"  top_transform: {top_tf}")
    print(f"  bot_transform: {bot_tf}")
    print(f"  scale ratio top/bot: {top_tf.scale / bot_tf.scale:.5f}")
