#!/usr/bin/env python3
"""
autograder.py
-------------
Cleans up scanned exam PDFs by:
  1. Auto-rotating pages to upright orientation (via Tesseract OSD)
  2. Removing blank/white pages
  3. Saving the result as a new PDF (pages copied losslessly via pikepdf)

Requirements:
    pip install pdf2image pytesseract numpy pikepdf Pillow
    apt install tesseract-ocr poppler-utils   # or brew install on macOS

Setup:
    cd /Users/joschka/Desktop/Programming/Auto-Grader
    source .venv/bin/activate

Usage:
    python autograder.py input.pdf output.pdf
    python autograder.py input.pdf output.pdf --dpi 300 --blank-threshold 250 --blank-std 6
"""

import argparse
import os
import sys
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


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_rotation(image: Image.Image) -> int:
    """
    Use Tesseract OSD to detect how many degrees CCW the page needs to be
    rotated to appear upright.

    Returns one of: 0, 90, 180, 270
    Returns 0 if detection fails or confidence is too low.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))

        if confidence < 2.0:
            return 0

        return angle

    except pytesseract.TesseractError:
        return 0


def is_blank_page(image: Image.Image,
                  mean_threshold: float = BLANK_MEAN_THRESHOLD,
                  std_threshold: float = BLANK_STD_THRESHOLD) -> bool:
    """
    Returns True if the page is essentially blank (all white or near-white).
    Uses both mean brightness and standard deviation of a grayscale version.
    A very high mean (bright) AND very low std (uniform) = blank.
    """
    gray_img = image if image.mode == "L" else image.convert("L")
    gray = np.array(gray_img, dtype=np.float32)
    mean = gray.mean()
    std = gray.std()
    return (mean >= mean_threshold) and (std <= std_threshold)


def _osd_worker(page_num: int, input_path: str, dpi: int) -> tuple[int, int]:
    """
    Worker for parallel OSD: renders a single page at full DPI and detects rotation.
    Returns (page_num, angle).
    """
    images = convert_from_path(input_path, dpi=dpi,
                               first_page=page_num, last_page=page_num)
    angle = detect_rotation(images[0])
    return (page_num, angle)


def process_pdf(input_path: str,
                output_path: str,
                analysis_dpi: int = ANALYSIS_DPI,
                blank_mean: float = BLANK_MEAN_THRESHOLD,
                blank_std: float = BLANK_STD_THRESHOLD) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"\nProcessing: {input_path}")
    print(f"Analysis DPI: {analysis_dpi}  |  Blank thresholds: mean≥{blank_mean}, std≤{blank_std}")

    # ------------------------------------------------------------------
    # Pass 1: Fast blank detection at low DPI
    # ------------------------------------------------------------------
    print(f"\nPass 1: Rendering all pages at {BLANK_DPI} DPI for blank detection...")
    low_res_pages = convert_from_path(str(input_path), dpi=BLANK_DPI,
                                          grayscale=True, thread_count=os.cpu_count() or 4)
    total_pages = len(low_res_pages)
    print(f"Total pages: {total_pages}")

    content_page_nums = []  # 1-indexed page numbers
    blank_page_nums = []

    for i, page_img in enumerate(low_res_pages):
        page_num = i + 1
        if is_blank_page(page_img, blank_mean, blank_std):
            blank_page_nums.append(page_num)
        else:
            content_page_nums.append(page_num)

    print(f"  → {len(blank_page_nums)} blank pages, {len(content_page_nums)} content pages")

    # Free low-res images
    del low_res_pages

    if not content_page_nums:
        print("WARNING: All pages were removed (all blank?). Nothing to save.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pass 2: Parallel OSD at full DPI (content pages only)
    # ------------------------------------------------------------------
    num_workers = min(os.cpu_count() or 4, len(content_page_nums))
    print(f"\nPass 2: Running OSD on {len(content_page_nums)} content pages "
          f"at {analysis_dpi} DPI ({num_workers} workers)...")

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

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\nResults:")
    for pn in sorted(rotation_map):
        angle = rotation_map[pn]
        status = f"rotate {angle}°" if angle != 0 else "ok"
        print(f"  Page {pn:>3}: {status}")
    for pn in sorted(blank_page_nums):
        print(f"  Page {pn:>3}: blank (removed)")

    # ------------------------------------------------------------------
    # Build output PDF with pikepdf (lossless)
    # ------------------------------------------------------------------
    src_pdf = pikepdf.open(str(input_path))
    out_pdf = pikepdf.new()

    for pn in content_page_nums:
        src_page = src_pdf.pages[pn - 1]  # 0-indexed
        angle = rotation_map.get(pn, 0)

        if angle != 0:
            try:
                existing_rotate = int(src_page.get("/Rotate", 0))
            except (TypeError, ValueError):
                existing_rotate = 0
            new_rotate = (existing_rotate + angle) % 360
            src_page["/Rotate"] = new_rotate

        out_pdf.pages.append(src_page)

    print(f"\nPages retained: {len(content_page_nums)}/{total_pages}")
    print(f"Saving to: {output_path}")

    out_pdf.save(str(output_path))
    out_pdf.close()
    src_pdf.close()

    print("Done.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-rotate and de-blank a scanned exam PDF."
    )
    parser.add_argument("input",  help="Path to input PDF")
    parser.add_argument("output", help="Path for output PDF")
    parser.add_argument("--dpi",  type=int,   default=ANALYSIS_DPI,
                        help=f"OSD analysis DPI (default: {ANALYSIS_DPI})")
    parser.add_argument("--blank-threshold", type=float, default=BLANK_MEAN_THRESHOLD,
                        help=f"Grayscale mean above which a page is blank (default: {BLANK_MEAN_THRESHOLD})")
    parser.add_argument("--blank-std", type=float, default=BLANK_STD_THRESHOLD,
                        help=f"Grayscale std below which a page is blank (default: {BLANK_STD_THRESHOLD})")
    args = parser.parse_args()

    process_pdf(
        input_path=args.input,
        output_path=args.output,
        analysis_dpi=args.dpi,
        blank_mean=args.blank_threshold,
        blank_std=args.blank_std,
    )


if __name__ == "__main__":
    main()
