"""Detect handwriting in yellow margin-strip regions of a deskewed scan PDF.

Step 13 classifies each yellow bbox as containing student handwriting (red overlay)
or blank (green overlay), producing cleaned_scan_refined_boxes.pdf.

Default detector: classical OpenCV — measures ink density after morphological line
removal. An optional PaddleOCR PPStructureV3 detector is available via
method="paddle" (requires paddle_env virtualenv).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

_GREEN: tuple[float, float, float] = (0.0, 0.75, 0.2)
_RED: tuple[float, float, float] = (0.9, 0.1, 0.1)
_YELLOW: tuple[float, float, float] = (1.0, 0.9, 0.0)
_TEAL: tuple[float, float, float] = (0.0, 0.52, 0.55)

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
    env = os.environ.copy()
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # Chinese HuggingFace mirror

    result = subprocess.run(
        [str(_PADDLE_PYTHON), str(_WORKER)] + [str(p) for p in crop_paths],
        stdout=subprocess.PIPE,  # capture JSON result
        stderr=None,             # let paddle logs print directly to terminal
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"paddle_worker failed (exit {result.returncode})")
    return json.loads(result.stdout)


def _vline_mask(binary_inv: np.ndarray) -> np.ndarray:
    """Return a mask of vertical line pixels in *binary_inv*.

    Two-step approach:
    1. MORPH_OPEN with a tall 1-px-wide kernel finds the centre column of each
       printed ruling line (height = ``max(h // 10, 20)`` px).
    2. Horizontal dilation with a 9-px-wide kernel expands the mask to cover
       the full width of the line, ensuring complete erasure.
    """
    kh = max(binary_inv.shape[0] // 10, 20)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    vlines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vkernel, iterations=2)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    return cv2.dilate(vlines, hkernel, iterations=1)


def _remove_vertical_lines(binary_inv: np.ndarray) -> np.ndarray:
    """Subtract the vertical line mask from *binary_inv*."""
    return cv2.subtract(binary_inv, _vline_mask(binary_inv))


def _has_handwriting_cv(
    crop_bgr: np.ndarray,
    *,
    ink_threshold: float = 0.003,
    min_blob_size: int = 30,
) -> bool:
    """Return True if *crop_bgr* contains handwriting after vertical line removal.

    Strategy:
    1. Convert to grayscale and binarise with Otsu thresholding.
    2. Remove printed vertical ruling lines via morphological MORPH_OPEN.
    3. Discard small noise blobs via connected-component filtering.
    4. Return True if the remaining ink density exceeds *ink_threshold*.

    Args:
        crop_bgr:       BGR crop of one yellow margin strip.
        ink_threshold:  Fraction of crop pixels that must remain after line
                        removal and noise filtering for a positive detection.
                        0.003 = 0.3 % of pixels.
        min_blob_size:  Connected-component area (px²) below which a blob is
                        treated as scanner noise and discarded.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = _remove_vertical_lines(binary_inv)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    filtered = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_size:
            filtered[labels == i] = 255
    ink_ratio = np.count_nonzero(filtered) / max(filtered.size, 1)
    return bool(ink_ratio > ink_threshold)


def detect_handwriting_in_rects(
    scan_pdf: Path,
    page_idx: int,
    rects: list[fitz.Rect],
    dpi: int,
    *,
    method: str = "classical",
    ink_threshold: float = 0.0007,
    min_blob_size: int = 15,
) -> list[HWResult]:
    """Rasterize one PDF page, crop each rect, detect handwriting.

    Args:
        scan_pdf:       Path to the deskewed scan PDF (cleaned_scan.pdf).
        page_idx:       Zero-based page index to rasterize.
        rects:          Yellow margin-strip rects in PDF points.
        dpi:            Raster DPI — must match the DPI used to build the transforms.
        method:         ``"classical"`` (default) — morphological vertical-line removal
                        + ink density check (fast, no external dependencies);
                        ``"paddle"`` — PaddleOCR PPStructureV3 via subprocess
                        (requires paddle_env virtualenv).
        ink_threshold:  Fraction of crop pixels that must remain after line removal
                        to count as handwriting. Lower = more sensitive.
        min_blob_size:  Minimum connected-component area (px²) to keep; smaller
                        blobs are treated as scanner noise. Lower = more sensitive.

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

    if method == "classical":
        results: list[HWResult] = [HWResult(r, has_handwriting=False) for r in rects]
        for i, rect in enumerate(rects):
            x0 = max(0, int(rect.x0 / px_to_pt))
            y0 = max(0, int(rect.y0 / px_to_pt))
            x1 = min(pix.width,  int(rect.x1 / px_to_pt))
            y1 = min(pix.height, int(rect.y1 / px_to_pt))
            crop = img_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            results[i] = HWResult(rect, has_handwriting=_has_handwriting_cv(
                crop, ink_threshold=ink_threshold, min_blob_size=min_blob_size
            ))
        return results

    # method == "paddle" — PaddleOCR via subprocess
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


def _erase_vertical_lines_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Return a copy of *crop_bgr* with detected vertical line pixels whitened."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = _vline_mask(binary_inv)
    result = crop_bgr.copy()
    result[mask > 0] = 255
    return result


def remove_vertical_lines_pdf(
    scan_pdf: Path, output_pdf: Path, *, dpi: int = 300
) -> Path:
    """Rasterise every page of *scan_pdf*, erase all structural vertical lines,
    and write the result to *output_pdf*.

    Unlike *write_vlines_removed_pdf*, which only erases lines inside yellow
    margin strips, this function processes the full page image so that all 6
    structural ruling lines (left margin, centre, right margin on both the top
    and bottom halves of a 4-up scan) are removed in one pass.

    Atomic write: saves to a temp file first, then replaces *output_pdf*.
    """
    doc_in = fitz.open(str(scan_pdf))
    doc_out = fitz.open()
    try:
        for page_in in doc_in:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page_in.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = _erase_vertical_lines_from_crop(img_bgr)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _, jpg_bytes = cv2.imencode(
                ".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            page_out = doc_out.new_page(
                width=page_in.rect.width, height=page_in.rect.height
            )
            page_out.insert_image(page_out.rect, stream=jpg_bytes.tobytes())

        tmp = output_pdf.with_suffix(".tmp.pdf")
        doc_out.save(str(tmp), garbage=4, deflate=True)
    finally:
        doc_in.close()
        doc_out.close()
    tmp.replace(output_pdf)
    return output_pdf


def write_vlines_removed_pdf(
    scan_pdf: Path,
    projected_boxes_json: Path,
    output_pdf: Path,
    page_results: dict[int, list[HWResult]],
    *,
    page_indices: tuple[int, ...] | None = None,
    line_width: float = 0.9,
) -> Path:
    """Write a copy of the scan with vertical lines erased inside yellow strips.

    For each processed page, rasterises the clean base scan, erases detected
    vertical ruling lines inside every yellow margin-strip region, then rebuilds
    the PDF page with all coloured overlays redrawn on top of the cleaned raster.
    The previously-yellow boxes are drawn green (no handwriting) or red (handwriting
    detected), using *page_results* from the handwriting detection step.

    Args:
        scan_pdf:             ``cleaned_scan.pdf`` — the deskewed base scan.
        projected_boxes_json: ``scan_projected_boxes.json`` from step 10.
        output_pdf:           Destination path.
        page_results:         Handwriting detection results keyed by page index.
        page_indices:         Zero-based page indices to process. ``None`` means
                              all pages present in the JSON.
        line_width:           Stroke width for the redrawn box overlays (points).

    Returns:
        *output_pdf* after saving.
    """
    payload = json.loads(Path(projected_boxes_json).read_text(encoding="utf-8"))
    dpi = int(payload.get("dpi", 300))
    px_to_pt = 72.0 / dpi
    pages_data: list[dict] = payload.get("pages") or []

    doc_in = fitz.open(str(scan_pdf))
    doc_out = fitz.open()
    try:
        for pd in pages_data:
            page_idx = int(pd["page_idx"])
            if page_indices is not None and page_idx not in page_indices:
                continue
            if page_idx >= len(doc_in):
                continue

            page_in = doc_in[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page_in.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for entry in pd.get("yellow", []):
                rx0, ry0, rx1, ry1 = entry["rect"]
                x0 = max(0, int(rx0 / px_to_pt))
                y0 = max(0, int(ry0 / px_to_pt))
                x1 = min(pix.width,  int(rx1 / px_to_pt))
                y1 = min(pix.height, int(ry1 / px_to_pt))
                if x1 > x0 and y1 > y0:
                    img_bgr[y0:y1, x0:x1] = _erase_vertical_lines_from_crop(
                        img_bgr[y0:y1, x0:x1]
                    )

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _, jpg_bytes = cv2.imencode(".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])

            page_out = doc_out.new_page(
                width=page_in.rect.width, height=page_in.rect.height
            )
            page_out.insert_image(page_out.rect, stream=jpg_bytes.tobytes())

            for entry in pd.get("exercise", []):
                page_out.draw_rect(
                    fitz.Rect(entry["rect"]),
                    color=tuple(entry["color"]),
                    width=line_width,
                )
            for entry in pd.get("eq_blank", []):
                page_out.draw_rect(
                    fitz.Rect(entry["rect"]), color=_TEAL, width=line_width
                )
            for hw in page_results.get(page_idx, []):
                color = _RED if hw.has_handwriting else _GREEN
                dashes = "[4 4] 0" if hw.has_handwriting else None
                page_out.draw_rect(hw.rect, color=color, width=line_width, dashes=dashes)

        doc_out.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        doc_in.close()
        doc_out.close()

    return output_pdf


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
            dashes = "[4 4] 0" if hw.has_handwriting else None
            page.draw_rect(hw.rect, color=color, width=line_width, dashes=dashes)
    doc.save(str(tmp), garbage=4, deflate=True)
    doc.close()
    tmp.replace(output_pdf)
    return output_pdf


# ---------------------------------------------------------------------------
# Adjusted exercise boxes
# ---------------------------------------------------------------------------

def compute_adjusted_exercise_boxes_for_page(
    page_data: dict,
    hw_results: list[HWResult],
) -> list[dict]:
    """Merge or trim each exercise box based on handwriting detection in its margin strip.

    For every exercise[i] / yellow[i] pair (same index, same vertical extent):
    - Handwriting detected → expand the exercise rect to include the margin strip.
    - No handwriting detected → keep the exercise rect unchanged (margin discarded).

    Args:
        page_data:   One page entry from ``scan_projected_boxes.json``
                     (must have ``"exercise"`` and ``"yellow"`` lists).
        hw_results:  Handwriting results for this page, one per yellow rect,
                     in the same order as the ``"yellow"`` list.

    Returns:
        List of dicts: ``{rect, color, expanded}`` — one per exercise box.
    """
    exercise_list = page_data.get("exercise", [])
    yellow_list = page_data.get("yellow", [])
    adjusted: list[dict] = []
    for ex_entry, yw_entry, hw in zip(exercise_list, yellow_list, hw_results):
        ex_rect = fitz.Rect(ex_entry["rect"])
        if hw.has_handwriting:
            yw_rect = fitz.Rect(yw_entry["rect"])
            merged = ex_rect.include_rect(yw_rect)
            adjusted.append({
                "rect": [merged.x0, merged.y0, merged.x1, merged.y1],
                "color": ex_entry["color"],
                "expanded": True,
            })
        else:
            adjusted.append({
                "rect": ex_entry["rect"],
                "color": ex_entry["color"],
                "expanded": False,
            })
    return adjusted


def write_adjusted_exercise_pdf(
    scan_pdf: Path,
    projected_boxes_json: Path,
    output_pdf: Path,
    adjusted_data: dict[int, list[dict]],
    *,
    dpi: int = 300,
    line_width: float = 0.9,
) -> Path:
    """Write a PDF showing only the adjusted exercise boxes on the cleaned scan.

    Identical base processing to ``write_vlines_removed_pdf`` (vertical lines
    erased inside yellow strips), but only the adjusted exercise boxes are drawn —
    no yellow overlays, no teal equation boxes.

    Args:
        scan_pdf:             ``cleaned_scan.pdf`` — deskewed base scan.
        projected_boxes_json: ``scan_projected_boxes.json`` from step 10.
        output_pdf:           Destination path.
        adjusted_data:        Mapping of page_idx → adjusted exercise entries
                              from ``compute_adjusted_exercise_boxes_for_page``.
        dpi:                  Raster DPI matching the scan.
        line_width:           Stroke width for exercise box outlines (points).

    Returns:
        *output_pdf* after atomic save.
    """
    payload = json.loads(Path(projected_boxes_json).read_text(encoding="utf-8"))
    dpi = int(payload.get("dpi", dpi))
    px_to_pt = 72.0 / dpi
    pages_data: list[dict] = payload.get("pages") or []

    doc_in = fitz.open(str(scan_pdf))
    doc_out = fitz.open()
    try:
        for pd in pages_data:
            page_idx = int(pd["page_idx"])
            if page_idx >= len(doc_in):
                continue

            page_in = doc_in[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page_in.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Erase vertical lines inside every original yellow strip
            for entry in pd.get("yellow", []):
                rx0, ry0, rx1, ry1 = entry["rect"]
                x0 = max(0, int(rx0 / px_to_pt))
                y0 = max(0, int(ry0 / px_to_pt))
                x1 = min(pix.width, int(rx1 / px_to_pt))
                y1 = min(pix.height, int(ry1 / px_to_pt))
                if x1 > x0 and y1 > y0:
                    img_bgr[y0:y1, x0:x1] = _erase_vertical_lines_from_crop(
                        img_bgr[y0:y1, x0:x1]
                    )

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _, jpg_bytes = cv2.imencode(".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])

            page_out = doc_out.new_page(
                width=page_in.rect.width, height=page_in.rect.height
            )
            page_out.insert_image(page_out.rect, stream=jpg_bytes.tobytes())

            # Draw only the adjusted exercise boxes for this page
            for entry in adjusted_data.get(page_idx, []):
                page_out.draw_rect(
                    fitz.Rect(entry["rect"]),
                    color=tuple(entry["color"]),
                    width=line_width,
                )

        doc_out.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        doc_in.close()
        doc_out.close()

    return output_pdf
