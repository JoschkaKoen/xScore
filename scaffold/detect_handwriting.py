"""Detect handwriting in yellow margin-strip regions of a deskewed scan PDF.

Step 11 uses this to classify each yellow bbox as containing student handwriting
(red overlay) or blank (green overlay), producing cleaned_scan_refined_boxes.pdf.

PaddleOCR runs in a dedicated paddle_env subprocess to avoid Python version conflicts.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

_GREEN: tuple[float, float, float] = (0.0, 0.75, 0.2)
_RED: tuple[float, float, float] = (0.9, 0.1, 0.1)

# paddle_env lives alongside the project root.
_PADDLE_PYTHON = Path(__file__).parent.parent / "paddle_env" / "bin" / "python"
_WORKER = Path(__file__).parent / "paddle_worker.py"


@dataclass
class HWResult:
    rect: fitz.Rect
    has_handwriting: bool


def _run_paddle_worker(crop_paths: list[Path]) -> list[bool]:
    """Call the paddle_worker subprocess with a batch of image paths.

    Returns one bool per path: True if handwriting was detected.
    Raises RuntimeError if the subprocess fails.
    """
    if not _PADDLE_PYTHON.exists():
        raise FileNotFoundError(
            f"paddle_env not found at {_PADDLE_PYTHON.parent.parent}. "
            "Create it with: python3 -m venv paddle_env && paddle_env/bin/pip install paddleocr"
        )
    result = subprocess.run(
        [str(_PADDLE_PYTHON), str(_WORKER)] + [str(p) for p in crop_paths],
        stdout=subprocess.PIPE,  # capture JSON result
        stderr=None,             # let paddle logs print directly to terminal
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"paddle_worker failed (exit {result.returncode})")
    return json.loads(result.stdout)


def detect_handwriting_in_rects(
    scan_pdf: Path,
    page_idx: int,
    rects: list[fitz.Rect],
    dpi: int,
) -> list[HWResult]:
    """Rasterize one PDF page, crop each rect, run PaddleOCR via subprocess.

    Args:
        scan_pdf:  Path to the deskewed scan PDF (cleaned_scan.pdf).
        page_idx:  Zero-based page index to rasterize.
        rects:     Yellow margin-strip rects in PDF points.
        dpi:       Raster DPI — must match the DPI used to build the transforms.

    Returns:
        One HWResult per input rect, in the same order.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    doc = fitz.open(str(scan_pdf))
    pix = doc[page_idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    px_to_pt = 72.0 / dpi
    crop_paths: list[Path] = []
    valid_indices: list[int] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, rect in enumerate(rects):
            x0 = max(0, int(rect.x0 / px_to_pt))
            y0 = max(0, int(rect.y0 / px_to_pt))
            x1 = min(pix.width, int(rect.x1 / px_to_pt))
            y1 = min(pix.height, int(rect.y1 / px_to_pt))
            crop = img_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_path = tmp_dir / f"crop_{i}.png"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(crop_path)
            valid_indices.append(i)

        hw_flags = _run_paddle_worker(crop_paths) if crop_paths else []

    results: list[HWResult] = [HWResult(rect, has_handwriting=False) for rect in rects]
    for flag, idx in zip(hw_flags, valid_indices):
        results[idx] = HWResult(rects[idx], has_handwriting=flag)
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
