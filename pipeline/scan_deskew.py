"""Automated fine deskew for scanned A3-portrait exam papers.

Each A3 page contains two A4 exam sheets (top half / bottom half). The scanner
introduces independent sub-degree skew in each half, so angle detection and
correction are performed **per half** and the halves are reassembled.

Angle detection uses vertical-projection variance on the **full-resolution**
grayscale half (no downsample); the correction uses the same angle with
bicubic interpolation on that same resolution.

Empirical data from 34 pages of a Space Physics scan (300 DPI, 68 half-pages):
  Range -0.45 to +0.20 deg, median -0.30 deg, 91 % of halves |skew| > 0.1 deg.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
_SWEEP_STEP = 0.05          # deg
_MIN_ABS_DEG = 0.05         # skip warp if detected angle is below this


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
# Per-page half-split pipeline
# ---------------------------------------------------------------------------

def deskew_page_halves(
    page_gray: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Split *page_gray* at the vertical midpoint, deskew each half separately.

    Returns:
        (deskewed_full_page, top_angle, bottom_angle)
    """
    h = page_gray.shape[0]
    mid = h // 2

    top = page_gray[:mid, :]
    bot = page_gray[mid:, :]

    top_angle = get_deskew_angle(top)
    bot_angle = get_deskew_angle(bot)

    top_fixed = deskew_image(top, top_angle)
    bot_fixed = deskew_image(bot, bot_angle)

    return np.vstack([top_fixed, bot_fixed]), top_angle, bot_angle


# ---------------------------------------------------------------------------
# Full PDF pipeline
# ---------------------------------------------------------------------------

def _process_page(args: tuple) -> tuple[int, Image.Image, float, float]:
    """Worker: deskew a single page (for use with ThreadPoolExecutor)."""
    page_idx, pil_img = args
    gray = np.array(pil_img.convert("L"))
    fixed_gray, top_angle, bot_angle = deskew_page_halves(gray)
    fixed_pil = Image.fromarray(fixed_gray, mode="L")
    return page_idx, fixed_pil, top_angle, bot_angle


def deskew_pdf_raster(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int = 300,
) -> Path:
    """Rasterize *input_pdf*, deskew each page (per half), write *output_pdf*.

    Args:
        input_pdf: Source PDF (already OSD-rotated by autograder).
        output_pdf: Destination PDF (may be the same path as input_pdf to overwrite).
        dpi: Render/output DPI.

    Returns:
        Path to the written output PDF.
    """
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    print(f"\n[deskew] Rendering {input_pdf.name} at {dpi} DPI …")
    print(
        "[deskew] Angle detection: full-resolution halves (no proxy downsample)"
    )
    pages = convert_from_path(
        str(input_pdf),
        dpi=dpi,
        grayscale=True,
        thread_count=os.cpu_count() or 4,
    )
    n = len(pages)
    print(f"[deskew] {n} pages loaded")

    results: dict[int, tuple[Image.Image, float, float]] = {}

    num_workers = min(os.cpu_count() or 4, n)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_process_page, (i, pages[i])): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            page_idx, fixed_pil, top_angle, bot_angle = fut.result()
            results[page_idx] = (fixed_pil, top_angle, bot_angle)
            print(
                f"[deskew]   page {page_idx + 1:>3}/{n}"
                f"  top={top_angle:+.2f}°  bot={bot_angle:+.2f}°"
            )

    # If output == input we need a temp file to avoid overwriting while still reading
    use_tmp = output_pdf.resolve() == input_pdf.resolve()
    write_target = output_pdf.with_suffix(".deskew_tmp.pdf") if use_tmp else output_pdf

    doc = fitz.open()
    for i in range(n):
        pil_img, _, _ = results[i]
        # Convert PIL image to PNG bytes
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        w_px, h_px = pil_img.size
        # PDF points at given DPI
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
