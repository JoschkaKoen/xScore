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

def detect_reference_lines(half_gray: np.ndarray) -> list[ReferenceLine]:
    """Locate the three vertical ruling lines on a deskewed A4 half-page.

    Uses morphological opening with a tall vertical kernel to erase handwriting
    and short printed elements, leaving only the tall Cambridge ruling lines.
    Segments within ``_MERGE_X_TOL`` px of each other in x are merged (handles
    the occasional 3-4 px gap in the right-hand column line on some pages).

    Args:
        half_gray: Grayscale uint8 array for one A4 half (top or bottom).

    Returns:
        List of ``ReferenceLine`` objects sorted by x_center.
        Logs a warning to stdout if a count other than 3 is found.
    """
    hh, hw = half_gray.shape[:2]

    # Binarise: ink → 255, paper → 0
    _, binary = cv2.threshold(half_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening: only structures taller than the kernel survive
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _VKERNEL_HEIGHT))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vkernel, iterations=2)

    # Connected-component analysis
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        v_mask, connectivity=8
    )

    # Collect blobs that look like full-height ruling lines
    raw: list[tuple[int, int, int]] = []  # (x_center, y_start, y_end)
    for i in range(1, num_labels):         # skip background label 0
        x, y, bw, bh, _area = stats[i]
        if bh > hh * _MIN_LINE_HEIGHT_FRAC and bw < _MAX_LINE_WIDTH:
            raw.append((int(x + bw // 2), int(y), int(y + bh)))

    # Merge segments with similar x positions (handles split right-hand line)
    raw.sort(key=lambda t: t[0])
    merged: list[ReferenceLine] = []
    for xc, ys, ye in raw:
        if merged and abs(xc - merged[-1].x_center) <= _MERGE_X_TOL:
            prev = merged[-1]
            merged[-1] = ReferenceLine(
                x_center=(prev.x_center + xc) // 2,
                y_start=min(prev.y_start, ys),
                y_end=max(prev.y_end, ye),
            )
        else:
            merged.append(ReferenceLine(x_center=xc, y_start=ys, y_end=ye))

    if len(merged) != 3:
        print(
            f"[deskew] WARNING: expected 3 reference lines, found {len(merged)} "
            f"(half size {hh}x{hw})"
        )

    return merged


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
