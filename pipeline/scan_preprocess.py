"""Rotate scanned exam PDFs upright (Tesseract OSD) and drop blank pages (pikepdf).

Used by :mod:`pipeline.pdf_cleanup` before fine deskew. Formerly ``autograder.py``.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pikepdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

ANALYSIS_DPI = 300          # DPI for rendering pages for OSD detection only.
BLANK_DPI = 72              # Low DPI for fast blank-page detection.
BLANK_MEAN_THRESHOLD = 250  # Pages with grayscale mean above this are considered blank (0-255)
BLANK_STD_THRESHOLD = 6     # Pages with grayscale std below this are considered blank


def detect_rotation(image: Image.Image) -> int:
    """Return CCW rotation (0, 90, 180, 270) needed for upright page, or 0 on failure."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))

        if confidence < 2.0:
            return 0

        return angle

    except pytesseract.TesseractError:
        return 0


def is_blank_page(
    image: Image.Image,
    mean_threshold: float = BLANK_MEAN_THRESHOLD,
    std_threshold: float = BLANK_STD_THRESHOLD,
) -> bool:
    gray_img = image if image.mode == "L" else image.convert("L")
    gray = np.array(gray_img, dtype=np.float32)
    mean = gray.mean()
    std = gray.std()
    return (mean >= mean_threshold) and (std <= std_threshold)


def _osd_worker(page_num: int, input_path: str, dpi: int) -> tuple[int, int]:
    images = convert_from_path(
        input_path, dpi=dpi, first_page=page_num, last_page=page_num
    )
    angle = detect_rotation(images[0])
    return (page_num, angle)


def process_pdf(
    input_path: str,
    output_path: str,
    analysis_dpi: int = ANALYSIS_DPI,
    blank_mean: float = BLANK_MEAN_THRESHOLD,
    blank_std: float = BLANK_STD_THRESHOLD,
    *,
    verbose: bool = True,
) -> None:
    """Blank detection + OSD rotation; write lossless PDF to *output_path*."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    from pipeline.terminal_ui import BOLD, CYAN, err_line, icon, note_line, ok_line, paint, warn_line

    if input_path.resolve() == output_path.resolve():
        err_line(
            "Input and output paths are the same — refusing to overwrite the source PDF. "
            "Choose a different output path."
        )
        sys.exit(1)

    if not input_path.exists():
        err_line(f"Input file not found: {input_path}")
        sys.exit(1)

    if verbose:
        print()
        print(paint(f"  {icon('doc')}  PDF prep  —  {input_path.name}", CYAN, BOLD))
        note_line(f"Full path: {input_path}")
        note_line(
            f"Analysis DPI: {analysis_dpi}  |  Blank: mean≥{blank_mean}, std≤{blank_std}"
        )
    else:
        note_line(
            f"PDF prep: {input_path.name}  |  OSD {analysis_dpi} DPI  |  "
            f"blank mean≥{blank_mean}, std≤{blank_std}"
        )

    if verbose:
        print(paint(f"\n  {icon('broom')}  Pass 1: blank detection @ {BLANK_DPI} DPI", CYAN, BOLD))
    else:
        note_line(f"Pass 1: blank detection @ {BLANK_DPI} DPI")
    low_res_pages = convert_from_path(
        str(input_path), dpi=BLANK_DPI, grayscale=True, thread_count=os.cpu_count() or 4
    )
    total_pages = len(low_res_pages)
    if verbose:
        print(f"Total pages: {total_pages}")
    else:
        note_line(f"{total_pages} pages scanned")

    content_page_nums: list[int] = []
    blank_page_nums: list[int] = []

    for i, page_img in enumerate(low_res_pages):
        page_num = i + 1
        if is_blank_page(page_img, blank_mean, blank_std):
            blank_page_nums.append(page_num)
        else:
            content_page_nums.append(page_num)

    if verbose:
        print(f"  → {len(blank_page_nums)} blank pages, {len(content_page_nums)} content pages")
    else:
        note_line(
            f"→ {len(blank_page_nums)} blank, {len(content_page_nums)} content pages "
            "(blank source pages are dropped from the output PDF)"
        )

    del low_res_pages

    if not content_page_nums:
        warn_line("All pages were removed (all blank?). Nothing to save.")
        sys.exit(1)

    num_workers = min(os.cpu_count() or 4, len(content_page_nums))
    if verbose:
        print(
            paint(
                f"\n  {icon('gear')}  Pass 2: OSD rotation on {len(content_page_nums)} pages "
                f"@ {analysis_dpi} DPI ({num_workers} workers)",
                CYAN,
                BOLD,
            )
        )
    else:
        note_line(
            f"Pass 2: OSD on {len(content_page_nums)} pages @ {analysis_dpi} DPI "
            f"({num_workers} workers)"
        )

    rotation_map: dict[int, int] = {}
    input_str = str(input_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_osd_worker, pn, input_str, analysis_dpi): pn
            for pn in content_page_nums
        }
        for future in as_completed(futures):
            page_num, angle = future.result()
            rotation_map[page_num] = angle

    if verbose:
        print(paint(f"\n  {icon('chart')}  Rotation summary", CYAN, BOLD))
        for pn in sorted(rotation_map):
            angle = rotation_map[pn]
            status = f"rotate {angle}°" if angle != 0 else "ok"
            print(f"  Page {pn:>3}: {status}")
        for pn in sorted(blank_page_nums):
            print(f"  Page {pn:>3}: blank (removed)")
    else:
        angle_counts = Counter(rotation_map.values())
        parts: list[str] = []
        for angle in sorted(angle_counts):
            n = angle_counts[angle]
            if angle == 0:
                parts.append(f"{n} upright")
            else:
                parts.append(f"{n} rotate {angle}°")
        note_line("Rotation: " + ", ".join(parts) if parts else "no content pages")

    src_pdf = pikepdf.open(str(input_path))
    out_pdf = pikepdf.new()

    for pn in content_page_nums:
        src_page = src_pdf.pages[pn - 1]
        angle = rotation_map.get(pn, 0)

        if angle != 0:
            try:
                existing_rotate = int(src_page.get("/Rotate", 0))
            except (TypeError, ValueError):
                existing_rotate = 0
            new_rotate = (existing_rotate + angle) % 360
            src_page["/Rotate"] = new_rotate

        out_pdf.pages.append(src_page)

    note_line(f"Pages retained: {len(content_page_nums)}/{total_pages}")
    note_line(f"Writing: {output_path}")

    out_pdf.save(str(output_path))
    out_pdf.close()
    src_pdf.close()

    ok_line("Passes 1–2 complete (rotate + de-blank).")
    if verbose:
        print()
