"""Detect handwriting in yellow margin-strip regions of a deskewed scan PDF.

Step 11 uses this to classify each yellow bbox as containing student handwriting
(red overlay) or blank (green overlay), producing cleaned_scan_refined_boxes.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from paddleocr import PPStructure

_GREEN: tuple[float, float, float] = (0.0, 0.75, 0.2)
_RED: tuple[float, float, float] = (0.9, 0.1, 0.1)


@dataclass
class HWResult:
    rect: fitz.Rect
    has_handwriting: bool


def make_engine() -> PPStructure:
    """Initialize the PPStructure layout engine.

    Expensive on first call (downloads models ~1–2 GB). Initialize once and
    reuse across all crops.
    """
    return PPStructure(layout=True, show_log=False, lang="en")


def detect_handwriting_in_rects(
    scan_pdf: Path,
    page_idx: int,
    rects: list[fitz.Rect],
    dpi: int,
    engine: PPStructure,
) -> list[HWResult]:
    """Rasterize one PDF page, crop each rect, run PPStructure, return results.

    Args:
        scan_pdf:  Path to the deskewed scan PDF (cleaned_scan.pdf).
        page_idx:  Zero-based page index to rasterize.
        rects:     Yellow margin-strip rects in PDF points (from
                   compute_yellow_rects_for_page).
        dpi:       Raster DPI — must match the DPI used to build the transforms.
        engine:    A PPStructure instance from make_engine().

    Returns:
        One HWResult per input rect, in the same order.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    doc = fitz.open(str(scan_pdf))
    # Force RGB (3 channels) so the numpy reshape and cv2 conversion are always valid.
    pix = doc[page_idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    px_to_pt = 72.0 / dpi
    results: list[HWResult] = []
    for rect in rects:
        x0 = max(0, int(rect.x0 / px_to_pt))
        y0 = max(0, int(rect.y0 / px_to_pt))
        x1 = min(pix.width, int(rect.x1 / px_to_pt))
        y1 = min(pix.height, int(rect.y1 / px_to_pt))
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            results.append(HWResult(rect, has_handwriting=False))
            continue
        hw = any(region["type"] == "handwriting" for region in engine(crop))
        results.append(HWResult(rect, has_handwriting=hw))
    return results


def overlay_refined_boxes(
    projected_pdf: Path,
    output_pdf: Path,
    page_results: dict[int, list[HWResult]],
    *,
    line_width: float = 1.5,
) -> Path:
    """Draw red/green refined boxes on a copy of the projected scaffold PDF.

    Args:
        projected_pdf: Source PDF (cleaned_scan_projected_boxes.pdf) — read-only.
        output_pdf:    Destination path (cleaned_scan_refined_boxes.pdf).
        page_results:  Mapping of page_idx → list[HWResult] to overlay.
        line_width:    Stroke width in PDF points (slightly thicker than the 0.9pt
                       yellow boxes so the result stands out visually).

    Returns:
        output_pdf path after the atomic save.
    """
    tmp = output_pdf.with_suffix(".tmp.pdf")
    doc = fitz.open(str(projected_pdf))
    for page_idx, results in page_results.items():
        page = doc[page_idx]
        for hw in results:
            color = _RED if hw.has_handwriting else _GREEN
            page.draw_rect(hw.rect, color=color, width=line_width)
    doc.save(str(tmp), garbage=4, deflate=True)
    doc.close()
    tmp.replace(output_pdf)
    return output_pdf
