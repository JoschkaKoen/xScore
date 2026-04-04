"""Automated fine deskew for scanned A3-portrait exam papers.

Each A3 page contains two A4 exam sheets (top half / bottom half). The scanner
introduces independent sub-degree skew in each half, so angle detection and
correction are performed **per half** and the halves are reassembled.

The sidecar ``<stem>_anchors.json`` (next to the output PDF by default, or set
via ``reflines_sidecar``) stores **IGCSE header anchor** positions per page and
**vertical ruling lines** per half (``top`` / ``bot`` arrays from
:func:`detect_reference_lines`, run after per-half deskew).  Older runs may still
have ``<stem>_reflines.json``; :func:`resolve_deskew_sidecar` finds either.  *input_pdf*
and *output_pdf* must differ — the source file is never overwritten in-place.

Empirical data from 34 pages of a Space Physics scan (300 DPI, 68 half-pages):
  Range -0.45 to +0.20 deg, median -0.30 deg, 91 % of halves |skew| > 0.1 deg.
"""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SWEEP_MIN = -3.0           # deg
_SWEEP_MAX = 3.0            # deg
_SWEEP_STEP = 0.01          # deg — fine pass only (full-resolution thresh)
_SWEEP_PROXY_SCALE = 4      # linear downsample for coarse pass only (4× → 16× fewer pixels/warp)
_SWEEP_COARSE_STEP = 0.1    # deg — coarse pass (safe on proxy; geometric res ~0.26° at 4×)
_SWEEP_FINE_HALF = 0.15     # deg — fine window ± this around coarse best (covers grid error)
_MIN_PROXY_DIM = 80         # px — if proxy smaller, run coarse sweep on full-res thresh instead
_MIN_ABS_DEG = 0.05         # skip warp if detected angle is below this

# Morphological line detection
_VKERNEL_HEIGHT = 150       # px — minimum height a blob must survive MORPH_OPEN
_MIN_LINE_HEIGHT_FRAC = 0.3 # blob must span > this fraction of the half-page height
_MAX_LINE_WIDTH = 30        # px — blobs wider than this are not vertical lines
_MERGE_X_TOL = 10           # px — x-distance within which segments are merged
_ENDPOINT_STRIP_HALF = 4    # px — half-width of column strip used for endpoint scan
_ENDPOINT_MIN_RUN = 8       # px — minimum consecutive ink rows to count as a real endpoint

# IGCSE anchor detection
_ANCHOR_SEARCH_HEIGHT = 350  # px — rows from top of each half-page to search
_ANCHOR_MIN_SCORE = 0.5      # TM_CCOEFF_NORMED threshold for accepting a match
_ANCHOR_TEMPLATE_PADDING = 8 # px — padding around OCR bbox when cropping template

# Overlay colours / sizes
_OVERLAY_ANCHOR_COLOR: tuple[float, float, float] = (1.0, 0.55, 0.0)  # orange
_OVERLAY_CROSSHAIR_SIZE_PT = 8.0  # pt — arm length of anchor crosshair markers


# ---------------------------------------------------------------------------
# Reference-line dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReferenceLine:
    """One detected vertical ruling line on a deskewed A4 half-page.

    All coordinates are in pixels relative to the top-left of the half-page
    image (y=0 is the top of that half, not the top of the full A3 page).
    """
    x_center: int   # horizontal centre of the line in pixels
    y_start: int    # topmost pixel of the detected blob
    y_end: int      # bottommost pixel of the detected blob

    def __str__(self) -> str:
        return f"x={self.x_center}  y={self.y_start}..{self.y_end}  h={self.y_end - self.y_start}"


@dataclass
class AnchorPoint:
    """Detected position of one IGCSE header label on a deskewed sub-page.

    Coordinates are in pixels relative to the top-left of the **half-page**
    image (i.e. y=0 is the top of that half, not the top of the full A3 page).
    """
    x: int          # center x of matched template in half-page pixel coords
    y: int          # center y of matched template in half-page pixel coords
    score: float    # template match confidence (TM_CCOEFF_NORMED, 0..1)

    def __str__(self) -> str:
        return f"({self.x},{self.y}) s={self.score:.2f}"


# ---------------------------------------------------------------------------
# Angle detection
# ---------------------------------------------------------------------------

def _best_angle_projection_variance(
    thresh: np.ndarray,
    angle_min: float,
    angle_max: float,
    angle_step: float,
) -> float:
    """Return angle in [*angle_min*, *angle_max*] that maximises column-sum variance."""
    h, w = thresh.shape[:2]
    cx, cy = w // 2, h // 2
    best_angle = 0.0
    best_var = -1.0
    for angle in np.arange(angle_min, angle_max + angle_step / 2, angle_step):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            thresh, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        proj = np.sum(rotated, axis=0, dtype=np.float64)
        v = float(np.var(proj))
        if v > best_var:
            best_var = v
            best_angle = float(angle)
    return best_angle


def get_deskew_angle(gray: np.ndarray) -> float:
    """Detect the skew angle of *gray* via vertical-projection variance.

    Two-stage sweep: a **coarse** pass (0.1° steps) on a 1/4-scale Otsu proxy for
    speed, then a **fine** pass (0.01° steps) on full-resolution Otsu within
    ± ``_SWEEP_FINE_HALF``° of the coarse winner so 0.01° accuracy matches
    the legacy single-pass behaviour.

    Args:
        gray: Grayscale uint8 numpy array (any size).

    Returns:
        Best rotation angle in degrees (positive = CCW).
        The caller should rotate by *-angle* to straighten the image.
    """
    h, w = gray.shape[:2]
    pw, ph = w // _SWEEP_PROXY_SCALE, h // _SWEEP_PROXY_SCALE

    if min(pw, ph) >= _MIN_PROXY_DIM:
        small = cv2.resize(gray, (pw, ph), interpolation=cv2.INTER_AREA)
        _, thresh_coarse = cv2.threshold(
            small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coarse_best = _best_angle_projection_variance(
            thresh_coarse, _SWEEP_MIN, _SWEEP_MAX, _SWEEP_COARSE_STEP
        )
    else:
        _, thresh_coarse = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coarse_best = _best_angle_projection_variance(
            thresh_coarse, _SWEEP_MIN, _SWEEP_MAX, _SWEEP_COARSE_STEP
        )

    _, thresh_full = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    fine_lo = max(_SWEEP_MIN, coarse_best - _SWEEP_FINE_HALF)
    fine_hi = min(_SWEEP_MAX, coarse_best + _SWEEP_FINE_HALF)
    return _best_angle_projection_variance(
        thresh_full, fine_lo, fine_hi, _SWEEP_STEP
    )


# ---------------------------------------------------------------------------
# Angle application
# ---------------------------------------------------------------------------

def deskew_image(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *gray* by *angle* degrees (positive = CCW) at full resolution.

    Uses bicubic interpolation with white (255) border fill.
    Returns the original array unchanged if ``abs(angle) < _MIN_ABS_DEG``.
    """
    if abs(angle) < _MIN_ABS_DEG:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# ---------------------------------------------------------------------------
# Reference-line detection
# ---------------------------------------------------------------------------

def _scan_column_endpoints(
    binary: np.ndarray,
    x_center: int,
    strip_half: int = _ENDPOINT_STRIP_HALF,
    min_run: int = _ENDPOINT_MIN_RUN,
) -> tuple[int, int]:
    """Find the first and last ink row in a narrow column strip of *binary*.

    Extracts columns ``[x_center - strip_half .. x_center + strip_half]``,
    collapses them with ``np.max`` to a 1-D row mask.  To avoid single stray
    pixels or scan-edge artefacts (e.g. page border at y=0) being mistaken for
    the true line endpoint, requires ``min_run`` consecutive ink rows before
    declaring an endpoint valid.  This is immune to gaps in the *middle* of the
    line caused by handwriting or print breaks.

    Returns:
        (y_start, y_end) — both 0-indexed row numbers inclusive.
    """
    hh, hw = binary.shape[:2]
    x0 = max(0, x_center - strip_half)
    x1 = min(hw, x_center + strip_half + 1)
    strip = binary[:, x0:x1]
    row_mask = (np.max(strip, axis=1) > 0).astype(np.uint8)  # 0/1 per row

    # Find y_start: first row of a run >= min_run consecutive ink rows
    y_start = 0
    for r in range(hh - min_run + 1):
        if row_mask[r:r + min_run].all():
            y_start = r
            break

    # Find y_end: last row of a run >= min_run consecutive ink rows (scan backwards)
    y_end = hh - 1
    for r in range(hh - 1, min_run - 2, -1):
        if row_mask[r - min_run + 1:r + 1].all():
            y_end = r
            break

    return y_start, y_end


def detect_reference_lines(half_gray: np.ndarray) -> list[ReferenceLine]:
    """Locate the three vertical ruling lines on a deskewed A4 half-page.

    Two-step strategy:
    1. **x position** — morphological opening + connected-component analysis,
       which reliably isolates the long printed vertical structures and gives a
       stable ``x_center`` for each line.
    2. **y endpoints** — for each found x_center, scan a narrow column strip
       (±``_ENDPOINT_STRIP_HALF`` px) in the *original* binary image (before
       opening) for the first and last ink row.  This is immune to mid-line
       gaps caused by handwriting or printing breaks that would otherwise clip
       the blob's bounding box.

    Args:
        half_gray: Grayscale uint8 array for one A4 half (top or bottom).

    Returns:
        List of ``ReferenceLine`` objects sorted by x_center.
        Logs a warning to stdout if a count other than 3 is found.
    """
    hh, hw = half_gray.shape[:2]

    # Binarise once; reuse for both morphology and endpoint scan
    _, binary = cv2.threshold(half_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ---- Step 1: find x_center of each line via morphological opening --------
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _VKERNEL_HEIGHT))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vkernel, iterations=2)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        v_mask, connectivity=8
    )

    raw_x: list[int] = []
    for i in range(1, num_labels):
        x, _y, bw, bh, _area = stats[i]
        if bh > hh * _MIN_LINE_HEIGHT_FRAC and bw < _MAX_LINE_WIDTH:
            raw_x.append(int(x + bw // 2))

    # Merge x positions that belong to the same physical line
    raw_x.sort()
    merged_x: list[int] = []
    for xc in raw_x:
        if merged_x and abs(xc - merged_x[-1]) <= _MERGE_X_TOL:
            merged_x[-1] = (merged_x[-1] + xc) // 2
        else:
            merged_x.append(xc)

    # ---- Step 2: scan column strip for true y_start / y_end ------------------
    lines: list[ReferenceLine] = []
    for xc in merged_x:
        y_start, y_end = _scan_column_endpoints(binary, xc)
        lines.append(ReferenceLine(x_center=xc, y_start=y_start, y_end=y_end))

    if len(lines) != 3:
        from shared.terminal_ui import tool_line

        tool_line(
            "deskew",
            f"WARNING: expected 3 reference lines, found {len(lines)} (half size {hh}x{hw})",
        )

    return lines


# ---------------------------------------------------------------------------
# IGCSE anchor detection
# ---------------------------------------------------------------------------

def extract_igcse_template(
    top_half_gray: np.ndarray,
    search_height: int = _ANCHOR_SEARCH_HEIGHT,
    padding: int = _ANCHOR_TEMPLATE_PADDING,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Bootstrap the IGCSE label template from the left sub-page of *top_half_gray*.

    Runs Tesseract OCR on the top-left search strip of the **first** page's top
    half to find the word "IGCSE", then returns that region (with padding) as
    the template used for fast ``cv2.matchTemplate`` on all subsequent pages.

    Args:
        top_half_gray: Grayscale uint8 array for the top half of scan page 1.
        search_height: Number of rows to search from the top of the half.
        padding: Pixels to add around the detected OCR bounding box.

    Returns:
        Cropped grayscale template as a uint8 numpy array.

    Raises:
        RuntimeError: If "IGCSE" cannot be found in the expected region.
    """
    hh, hw = top_half_gray.shape[:2]
    mid_x = hw // 2
    # Search only the left sub-page header strip
    strip = top_half_gray[:min(search_height, hh), :mid_x]

    data = pytesseract.image_to_data(
        Image.fromarray(strip),
        output_type=pytesseract.Output.DICT,
    )

    best_conf = -1
    best_bbox: tuple[int, int, int, int] | None = None
    for i, text in enumerate(data["text"]):
        if "IGCSE" in text.upper():
            conf = int(data["conf"][i])
            if conf > best_conf:
                best_conf = conf
                best_bbox = (
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["left"][i]) + int(data["width"][i]),
                    int(data["top"][i]) + int(data["height"][i]),
                )

    if best_bbox is None:
        raise RuntimeError(
            "[deskew] Could not locate 'IGCSE' in the top-left header region of page 1. "
            "Ensure the scan is correctly oriented and the header is not obscured."
        )

    x0 = max(0, best_bbox[0] - padding)
    y0 = max(0, best_bbox[1] - padding)
    x1 = min(strip.shape[1], best_bbox[2] + padding)
    y1 = min(strip.shape[0], best_bbox[3] + padding)

    template = strip[y0:y1, x0:x1].copy()
    if verbose:
        from shared.terminal_ui import tool_line

        tool_line(
            "deskew",
            f"IGCSE template: {template.shape[1]}x{template.shape[0]}px "
            f"at bbox=({x0},{y0},{x1},{y1})  OCR conf={best_conf}",
        )
    return template


def detect_igcse_anchors(
    half_gray: np.ndarray,
    template: np.ndarray,
    search_height: int = _ANCHOR_SEARCH_HEIGHT,
    min_score: float = _ANCHOR_MIN_SCORE,
) -> tuple[AnchorPoint | None, AnchorPoint | None]:
    """Locate the IGCSE header label on the left and right sub-pages of *half_gray*.

    Uses ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED`` inside a restricted
    search region — the top *search_height* rows of each left/right half —
    to avoid false positives from scattered "IGCSE" labels further down the page.

    All returned coordinates are in **half-page pixel space** (y=0 is the top of
    this half, not the top of the full A3 page).

    Args:
        half_gray: Grayscale uint8 array for one A4 half (top or bottom).
        template: Template cropped by ``extract_igcse_template``.
        search_height: Rows from the top of the half-page to restrict search.
        min_score: Minimum ``TM_CCOEFF_NORMED`` score; matches below this are
            discarded and ``None`` is returned for that side.

    Returns:
        ``(left_anchor, right_anchor)`` — either may be ``None`` if no confident
        match is found.
    """
    hh, hw = half_gray.shape[:2]
    mid_x = hw // 2
    th, tw = template.shape[:2]
    search_h = min(search_height, hh)

    def _match_in(region: np.ndarray, x_offset: int) -> AnchorPoint | None:
        if region.shape[0] < th or region.shape[1] < tw:
            return None
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < min_score:
            return None
        # max_loc is the top-left corner of the best-matching patch; anchor = center
        ax = x_offset + max_loc[0] + tw // 2
        ay = max_loc[1] + th // 2
        return AnchorPoint(x=int(ax), y=int(ay), score=round(float(max_val), 3))

    left_anchor  = _match_in(half_gray[:search_h, :mid_x],  x_offset=0)
    right_anchor = _match_in(half_gray[:search_h, mid_x:],  x_offset=mid_x)

    if left_anchor is None or right_anchor is None:
        from shared.terminal_ui import tool_line

        if left_anchor is None:
            tool_line("deskew", "WARNING: IGCSE anchor not found in left sub-page header")
        if right_anchor is None:
            tool_line("deskew", "WARNING: IGCSE anchor not found in right sub-page header")

    return left_anchor, right_anchor


# ---------------------------------------------------------------------------
# Per-page half-split pipeline
# ---------------------------------------------------------------------------

def deskew_page_halves(
    page_gray: np.ndarray,
) -> tuple[np.ndarray, float, float, list[ReferenceLine], list[ReferenceLine]]:
    """Split *page_gray* at the vertical midpoint, deskew each half separately.

    After deskew, runs :func:`detect_reference_lines` on each half to locate the
    printed vertical ruling lines (typically three per half).

    Returns:
        (deskewed_full_page, top_angle, bot_angle, top_lines, bot_lines)
    """
    h = page_gray.shape[0]
    mid = h // 2

    top = page_gray[:mid, :]
    bot = page_gray[mid:, :]

    top_angle = get_deskew_angle(top)
    bot_angle = get_deskew_angle(bot)

    top_fixed = deskew_image(top, top_angle)
    bot_fixed = deskew_image(bot, bot_angle)

    top_lines = detect_reference_lines(top_fixed)
    bot_lines = detect_reference_lines(bot_fixed)

    return np.vstack([top_fixed, bot_fixed]), top_angle, bot_angle, top_lines, bot_lines


# ---------------------------------------------------------------------------
# Full PDF pipeline
# ---------------------------------------------------------------------------

_PageResult = tuple[Image.Image, float, float, list[ReferenceLine], list[ReferenceLine]]


def _process_page(
    args: tuple,
) -> tuple[int, Image.Image, float, float, list[ReferenceLine], list[ReferenceLine]]:
    """Worker: deskew a single page and detect reference lines."""
    page_idx, pil_img = args
    gray = np.array(pil_img.convert("L"))
    fixed_gray, top_angle, bot_angle, top_lines, bot_lines = deskew_page_halves(gray)
    fixed_pil = Image.fromarray(fixed_gray, mode="L")
    return page_idx, fixed_pil, top_angle, bot_angle, top_lines, bot_lines


def _lines_str(lines: list[ReferenceLine]) -> str:
    """Compact one-line summary of up to 3 detected reference lines."""
    labels = ["L", "C", "R"]
    parts = []
    for label, ln in zip(labels, lines):
        parts.append(f"{label}({ln.x_center},{ln.y_start}..{ln.y_end})")
    return "  ".join(parts) if parts else "(none)"


def anchors_sidecar_path(deskewed_pdf: Path) -> Path:
    """Path for the IGCSE anchor sidecar next to a deskewed raster PDF."""
    p = Path(deskewed_pdf)
    return p.with_name(p.stem + "_anchors.json")


def resolve_deskew_sidecar(deskewed_pdf: Path) -> Path | None:
    """Return an existing anchor sidecar path, or *None*.

    Prefers ``<stem>_anchors.json``; falls back to legacy ``<stem>_reflines.json``.
    """
    p = Path(deskewed_pdf)
    newer = anchors_sidecar_path(p)
    if newer.is_file():
        return newer
    legacy = p.with_name(p.stem + "_reflines.json")
    if legacy.is_file():
        return legacy
    return None


def deskew_pdf_raster(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int = 300,
    *,
    reflines_sidecar: Path | None = None,
    verbose: bool = True,
    saved_as: str | None = None,
) -> Path:
    """Rasterize *input_pdf*, deskew each page (per half), detect IGCSE anchors,
    write *output_pdf* and a sidecar JSON file (``*_anchors.json`` by default).

    **Input and output paths must differ** — raw or intermediate PDFs must never
    be overwritten in-place (the pipeline reads the whole file while building
    the new document).  Callers that want to replace an existing file should write
    to a sibling temp path and ``Path.replace`` afterward.

    Args:
        input_pdf: Source PDF (already OSD-rotated by :func:`preprocessing.remove_blanks_autorotate.process_pdf`).
        output_pdf: Destination PDF (must not resolve to the same path as *input_pdf*).
        dpi: Render/output DPI.
        reflines_sidecar: Optional path for the anchor sidecar JSON (IGCSE anchors
            per page).  Defaults to :func:`anchors_sidecar_path` applied to
            *output_pdf*.  Use this when *output_pdf* is a temp file but the sidecar
            should use the final stem (e.g. ``cleaned_scan_anchors.json``).
        verbose: When False (e.g. ``grade.py`` pipeline), print only summaries instead
            of per-page angle, line, and anchor lines.
        saved_as: If set, compact-mode success line shows this filename (e.g. final
            ``cleaned_scan.pdf``) when *output_pdf* is a temp path.

    Returns:
        Path to the written output PDF.
    """
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    in_r = input_pdf.resolve()
    out_r = output_pdf.resolve()
    if in_r == out_r:
        raise ValueError(
            "[deskew] input_pdf and output_pdf must be different paths — "
            "refusing to overwrite the source PDF. Write to a temp file, "
            "then Path.replace() if you need to update the original path."
        )

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

    from shared.terminal_ui import get_console, note_line, ok_line, progress_line, tool_line

    if verbose:
        get_console().print()
        tool_line("deskew", f"Rendering {input_pdf.name} at {dpi} DPI …")
        tool_line(
            "deskew",
            "Angle detection: coarse 0.1° on ¼-scale proxy, fine 0.01° at full resolution",
        )
    else:
        progress_line(
            "Deskew: rasterize → per-half skew → vertical ref-lines → IGCSE anchors → PDF (slow) …",
        )
    pages = convert_from_path(
        str(input_pdf),
        dpi=dpi,
        grayscale=True,
        thread_count=os.cpu_count() or 4,
    )
    n = len(pages)
    num_workers = min(os.cpu_count() or 4, n)
    if verbose:
        tool_line("deskew", f"{n} pages loaded")

    results: dict[int, _PageResult] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_process_page, (i, pages[i])): i
            for i in range(n)
        }
        if verbose:
            for fut in as_completed(futures):
                page_idx, fixed_pil, top_angle, bot_angle, top_lines, bot_lines = fut.result()
                results[page_idx] = (fixed_pil, top_angle, bot_angle, top_lines, bot_lines)
                tool_line(
                    "deskew",
                    f"  page {page_idx + 1:>3}/{n}"
                    f"  top={top_angle:+.2f}°  bot={bot_angle:+.2f}°",
                )
                tool_line("deskew", f"    top lines: {_lines_str(top_lines)}")
                tool_line("deskew", f"    bot lines: {_lines_str(bot_lines)}")
        else:
            with Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=get_console(),
                transient=True,
            ) as prog:
                task_id = prog.add_task("Deskew pages", total=n)
                for fut in as_completed(futures):
                    page_idx, fixed_pil, top_angle, bot_angle, top_lines, bot_lines = fut.result()
                    results[page_idx] = (fixed_pil, top_angle, bot_angle, top_lines, bot_lines)
                    prog.advance(task_id)

    # Bootstrap IGCSE template from page 0 top half (Tesseract, runs once)
    if verbose:
        tool_line("deskew", "Extracting IGCSE template from page 1 …")
    page0_gray = np.array(results[0][0].convert("L"))
    p0_mid = page0_gray.shape[0] // 2
    igcse_template = extract_igcse_template(page0_gray[:p0_mid, :], verbose=verbose)

    # Detect IGCSE anchors on all pages (fast template matching, serial)
    if verbose:
        tool_line("deskew", "Detecting IGCSE anchors …")
    page_anchors: dict[int, dict[str, AnchorPoint | None]] = {}
    for i in range(n):
        page_gray = np.array(results[i][0].convert("L"))
        p_mid = page_gray.shape[0] // 2
        tl, tr = detect_igcse_anchors(page_gray[:p_mid, :], igcse_template)
        bl, br = detect_igcse_anchors(page_gray[p_mid:, :], igcse_template)
        page_anchors[i] = {"top_left": tl, "top_right": tr, "bot_left": bl, "bot_right": br}
        if verbose:
            tool_line(
                "deskew",
                f"  page {i + 1:>3} anchors:"
                f"  TL={tl}  TR={tr}  BL={bl}  BR={br}",
            )
    # Build sidecar JSON (ordered by page index)
    def _anc_dict(a: AnchorPoint | None) -> dict | None:
        return asdict(a) if a is not None else None

    reflines_data: list[dict] = []
    for i in range(n):
        _, _, _, top_lines, bot_lines = results[i]
        anc = page_anchors[i]
        reflines_data.append({
            "page": i + 1,
            "top": [asdict(ln) for ln in top_lines],
            "bot": [asdict(ln) for ln in bot_lines],
            "anchors": {
                "top_left":  _anc_dict(anc["top_left"]),
                "top_right": _anc_dict(anc["top_right"]),
                "bot_left":  _anc_dict(anc["bot_left"]),
                "bot_right": _anc_dict(anc["bot_right"]),
            },
        })

    sidecar_path = (
        Path(reflines_sidecar).resolve()
        if reflines_sidecar is not None
        else anchors_sidecar_path(output_pdf).resolve()
    )
    sidecar_path.write_text(json.dumps(reflines_data, indent=2))

    if verbose:
        tool_line(
            "deskew",
            f"Sidecar (anchors + vertical ref-lines) → {sidecar_path.name}",
        )

    doc = fitz.open()
    for i in range(n):
        pil_img, *_ = results[i]
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        w_px, h_px = pil_img.size
        pt_per_px = 72.0 / dpi
        page = doc.new_page(width=w_px * pt_per_px, height=h_px * pt_per_px)
        rect = fitz.Rect(0, 0, w_px * pt_per_px, h_px * pt_per_px)
        page.insert_image(rect, stream=png_bytes)

    doc.save(str(output_pdf), deflate=True)
    doc.close()

    out_label = saved_as if saved_as is not None else output_pdf.name
    if not verbose:
        tops = [results[i][1] for i in range(n)]
        bots = [results[i][2] for i in range(n)]
        ref_ok = sum(
            1
            for i in range(n)
            if len(results[i][3]) == 3 and len(results[i][4]) == 3
        )
        ok_line(
            f"Deskew: {n}p @ {dpi} DPI · "
            f"skew top [{min(tops):+.2f}…{max(tops):+.2f}]° "
            f"bot [{min(bots):+.2f}…{max(bots):+.2f}]° · "
            f"vertical ref-lines 3+3 on {ref_ok}/{n} pages · "
            f"sidecar {sidecar_path.name} · PDF → {out_label}"
        )
    else:
        ref_ok = sum(
            1
            for i in range(n)
            if len(results[i][3]) == 3 and len(results[i][4]) == 3
        )
        ok_line(
            f"Deskew saved → {output_pdf.name} · "
            f"vertical ref-lines 3+3 on {ref_ok}/{n} pages"
        )
    return output_pdf


# ---------------------------------------------------------------------------
# Visualise detected reference lines on a deskewed raster PDF
# ---------------------------------------------------------------------------

# Default overlay stroke: vivid pink (RGB 0–1)
_OVERLAY_PINK: tuple[float, float, float] = (1.0, 0.35, 0.78)
_OVERLAY_LINE_WIDTH_PT = 0.35


def overlay_reflines_on_pdf(
    deskewed_pdf: Path,
    reflines_json: Path,
    output_pdf: Path,
    dpi: int = 300,
    line_rgb: tuple[float, float, float] = _OVERLAY_PINK,
    line_width_pt: float = _OVERLAY_LINE_WIDTH_PT,
) -> Path:
    """Draw vertical reference lines from ``top`` / ``bot`` (pink, if present)
    and IGCSE anchor crosshairs on a **copy** of *deskewed_pdf*, saved to
    *output_pdf*.  When those arrays are empty, only anchors are drawn.

    Coordinates in the JSON are pixel offsets on each A4 **half** (top: y from
    page top; bottom: y from the half-page boundary). This matches
    ``deskew_page_halves`` / ``deskew_pdf_raster``.
    """
    deskewed_pdf = Path(deskewed_pdf)
    reflines_json = Path(reflines_json)
    output_pdf = Path(output_pdf)

    data: list[dict] = json.loads(reflines_json.read_text())
    px_to_pt = 72.0 / dpi

    from shared.terminal_ui import tool_line

    doc = fitz.open(str(deskewed_pdf))
    try:
        if len(data) != len(doc):
            tool_line(
                "reflines_overlay",
                f"WARNING: JSON has {len(data)} pages, PDF has {len(doc)} — overlaying min length",
            )

        for entry in data:
            idx = int(entry["page"]) - 1
            if idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            h_px = int(round(page.rect.height / px_to_pt))
            mid = h_px // 2

            for ln in entry.get("top", []):
                xc = int(ln["x_center"])
                y0 = int(ln["y_start"])
                y1 = int(ln["y_end"])
                x_pt = xc * px_to_pt
                page.draw_line(
                    fitz.Point(x_pt, y0 * px_to_pt),
                    fitz.Point(x_pt, y1 * px_to_pt),
                    color=line_rgb,
                    width=line_width_pt,
                    lineCap=1,  # round caps
                )

            for ln in entry.get("bot", []):
                xc = int(ln["x_center"])
                y0 = mid + int(ln["y_start"])
                y1 = mid + int(ln["y_end"])
                x_pt = xc * px_to_pt
                page.draw_line(
                    fitz.Point(x_pt, y0 * px_to_pt),
                    fitz.Point(x_pt, y1 * px_to_pt),
                    color=line_rgb,
                    width=line_width_pt,
                    lineCap=1,
                )

            # Draw crosshair markers at each IGCSE anchor position
            anchors = entry.get("anchors", {})
            for key, half_offset_px in [
                ("top_left", 0), ("top_right", 0),
                ("bot_left", mid), ("bot_right", mid),
            ]:
                anc = anchors.get(key)
                if anc is None:
                    continue
                ax_pt = int(anc["x"]) * px_to_pt
                ay_pt = (half_offset_px + int(anc["y"])) * px_to_pt
                arm = _OVERLAY_CROSSHAIR_SIZE_PT
                page.draw_line(
                    fitz.Point(ax_pt - arm, ay_pt),
                    fitz.Point(ax_pt + arm, ay_pt),
                    color=_OVERLAY_ANCHOR_COLOR,
                    width=0.5,
                )
                page.draw_line(
                    fitz.Point(ax_pt, ay_pt - arm),
                    fitz.Point(ax_pt, ay_pt + arm),
                    color=_OVERLAY_ANCHOR_COLOR,
                    width=0.5,
                )

        doc.save(str(output_pdf), deflate=True)
    finally:
        doc.close()

    tool_line("reflines_overlay", f"Saved → {output_pdf}")
    return output_pdf
