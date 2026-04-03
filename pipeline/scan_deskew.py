"""Automated fine deskew for scanned A3-portrait exam papers.

Each A3 page contains two A4 exam sheets (top half / bottom half). The scanner
introduces independent sub-degree skew in each half, so angle detection and
correction are performed **per half** and the halves are reassembled.

After deskewing, the three vertical reference lines printed on each Cambridge
exam sheet (left edge, centre column, right edge) are located by morphological
opening with a 1×150 vertical kernel, which erases text/handwriting while
preserving the tall printed ruling lines.  Detected positions are stored in a
sidecar ``<output_stem>_reflines.json`` written next to the output PDF.

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
from pdf2image import convert_from_path
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SWEEP_MIN = -3.0           # deg
_SWEEP_MAX = 3.0            # deg
_SWEEP_STEP = 0.01          # deg
_MIN_ABS_DEG = 0.05         # skip warp if detected angle is below this

# Morphological line detection
_VKERNEL_HEIGHT = 150       # px — minimum height a blob must survive MORPH_OPEN
_MIN_LINE_HEIGHT_FRAC = 0.3 # blob must span > this fraction of the half-page height
_MAX_LINE_WIDTH = 30        # px — blobs wider than this are not vertical lines
_MERGE_X_TOL = 10           # px — x-distance within which segments are merged
_ENDPOINT_STRIP_HALF = 4    # px — half-width of column strip used for endpoint scan
_ENDPOINT_MIN_RUN = 8       # px — minimum consecutive ink rows to count as a real endpoint


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


# ---------------------------------------------------------------------------
# Angle detection
# ---------------------------------------------------------------------------

def get_deskew_angle(gray: np.ndarray) -> float:
    """Detect the skew angle of *gray* via vertical-projection variance.

    Runs Otsu binarisation and the angle sweep at **native resolution** of *gray*
    (no downsample), so each warp matches the pixels used for deskew_image.

    Args:
        gray: Grayscale uint8 numpy array (any size).

    Returns:
        Best rotation angle in degrees (positive = CCW).
        The caller should rotate by *-angle* to straighten the image.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape[:2]
    cx, cy = w // 2, h // 2
    best_angle = 0.0
    best_var = -1.0

    for angle in np.arange(_SWEEP_MIN, _SWEEP_MAX + _SWEEP_STEP / 2, _SWEEP_STEP):
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
        print(
            f"[deskew] WARNING: expected 3 reference lines, found {len(lines)} "
            f"(half size {hh}x{hw})"
        )

    return lines


# ---------------------------------------------------------------------------
# Per-page half-split pipeline
# ---------------------------------------------------------------------------

def deskew_page_halves(
    page_gray: np.ndarray,
) -> tuple[np.ndarray, float, float, list[ReferenceLine], list[ReferenceLine]]:
    """Split *page_gray* at the vertical midpoint, deskew each half separately,
    then detect the three vertical reference lines on each corrected half.

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


def deskew_pdf_raster(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int = 300,
) -> Path:
    """Rasterize *input_pdf*, deskew each page (per half), detect reference
    lines, write *output_pdf* and a sidecar ``<stem>_reflines.json``.

    Args:
        input_pdf: Source PDF (already OSD-rotated by autograder).
        output_pdf: Destination PDF (may equal input_pdf to overwrite in-place).
        dpi: Render/output DPI.

    Returns:
        Path to the written output PDF.
    """
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    print(f"\n[deskew] Rendering {input_pdf.name} at {dpi} DPI …")
    print("[deskew] Angle detection: full-resolution halves (no proxy downsample)")
    pages = convert_from_path(
        str(input_pdf),
        dpi=dpi,
        grayscale=True,
        thread_count=os.cpu_count() or 4,
    )
    n = len(pages)
    print(f"[deskew] {n} pages loaded")

    results: dict[int, _PageResult] = {}

    num_workers = min(os.cpu_count() or 4, n)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_process_page, (i, pages[i])): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            page_idx, fixed_pil, top_angle, bot_angle, top_lines, bot_lines = fut.result()
            results[page_idx] = (fixed_pil, top_angle, bot_angle, top_lines, bot_lines)
            print(
                f"[deskew]   page {page_idx + 1:>3}/{n}"
                f"  top={top_angle:+.2f}°  bot={bot_angle:+.2f}°"
            )
            print(f"[deskew]     top lines: {_lines_str(top_lines)}")
            print(f"[deskew]     bot lines: {_lines_str(bot_lines)}")

    # Build sidecar JSON (ordered by page index)
    reflines_data: list[dict] = []
    for i in range(n):
        _, _, _, top_lines, bot_lines = results[i]
        reflines_data.append({
            "page": i + 1,
            "top": [asdict(ln) for ln in top_lines],
            "bot": [asdict(ln) for ln in bot_lines],
        })

    sidecar_path = output_pdf.with_name(output_pdf.stem + "_reflines.json")
    sidecar_path.write_text(json.dumps(reflines_data, indent=2))
    print(f"[deskew] Reference lines → {sidecar_path.name}")

    # If output == input we need a temp file to avoid overwriting while reading
    use_tmp = output_pdf.resolve() == input_pdf.resolve()
    write_target = output_pdf.with_suffix(".deskew_tmp.pdf") if use_tmp else output_pdf

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

    doc.save(str(write_target), deflate=True)
    doc.close()

    if use_tmp:
        write_target.replace(output_pdf)

    print(f"[deskew] Saved → {output_pdf}")
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
    """Draw detected vertical reference lines (from *reflines_json*) in pink on a
    **copy** of *deskewed_pdf*, saved to *output_pdf*.

    Coordinates in the JSON are pixel offsets on each A4 **half** (top: y from
    page top; bottom: y from the half-page boundary). This matches
    ``deskew_page_halves`` / ``deskew_pdf_raster``.
    """
    deskewed_pdf = Path(deskewed_pdf)
    reflines_json = Path(reflines_json)
    output_pdf = Path(output_pdf)

    data: list[dict] = json.loads(reflines_json.read_text())
    px_to_pt = 72.0 / dpi

    doc = fitz.open(str(deskewed_pdf))
    try:
        if len(data) != len(doc):
            print(
                f"[reflines_overlay] WARNING: JSON has {len(data)} pages, "
                f"PDF has {len(doc)} — overlaying min length"
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

        doc.save(str(output_pdf), deflate=True)
    finally:
        doc.close()

    print(f"[reflines_overlay] Saved → {output_pdf}")
    return output_pdf
