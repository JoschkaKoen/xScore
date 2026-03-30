#!/usr/bin/env python3
"""
fix_scanned_pdf.py
------------------
Cleans up scanned exam PDFs by:
  1. Auto-rotating pages to upright orientation (via Tesseract OSD)
  2. Removing blank/white pages
  3. Saving the result as a new PDF (preserving original image data)

Requirements:
    pip install pdf2image pytesseract numpy pikepdf Pillow
    apt install tesseract-ocr poppler-utils   # or brew install on macOS


    cd /Users/joschka/Desktop/Programming/Auto-Grader
    source .venv/bin/activate
    python autograder.py input.pdf output.pdf

Usage:
    python fix_scanned_pdf.py input.pdf output.pdf
    python fix_scanned_pdf.py input.pdf output.pdf --dpi 300 --blank-threshold 250 --blank-std 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pikepdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

ANALYSIS_DPI = 300          # DPI for rendering pages for OSD / blank detection only.
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
            print(f"    OSD confidence too low ({confidence:.1f}), keeping as-is")
            return 0

        print(f"    OSD: rotate={angle}°, confidence={confidence:.1f}")
        return angle

    except pytesseract.TesseractError as e:
        print(f"    OSD failed ({e}), keeping as-is")
        return 0


def is_blank_page(image: Image.Image,
                  mean_threshold: float = BLANK_MEAN_THRESHOLD,
                  std_threshold: float = BLANK_STD_THRESHOLD) -> bool:
    """
    Returns True if the page is essentially blank (all white or near-white).
    Uses both mean brightness and standard deviation of a grayscale version.
    A very high mean (bright) AND very low std (uniform) = blank.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean = gray.mean()
    std = gray.std()
    blank = (mean >= mean_threshold) and (std <= std_threshold)
    print(f"    Pixel stats: mean={mean:.1f}, std={std:.1f} → {'BLANK' if blank else 'content'}")
    return blank


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
    print(f"Analysis DPI: {analysis_dpi}  |  Blank thresholds: mean≥{blank_mean}, std≤{blank_std}\n")

    # Open the original PDF — we will copy/rotate pages without re-encoding
    src_pdf = pikepdf.open(str(input_path))
    total_pages = len(src_pdf.pages)

    # Render pages at analysis DPI only for blank/orientation detection
    rendered = convert_from_path(str(input_path), dpi=analysis_dpi)
    print(f"Total pages in input: {total_pages}\n")

    out_pdf = pikepdf.new()
    kept = 0

    for i, page_img in enumerate(rendered):
        page_num = i + 1
        print(f"Page {page_num}/{total_pages}")

        # --- Blank page check ---
        if is_blank_page(page_img, blank_mean, blank_std):
            print(f"  → Removing blank page\n")
            continue

        # --- Orientation detection ---
        angle = detect_rotation(page_img)

        # Copy the original page (preserves embedded images byte-for-byte)
        src_page = src_pdf.pages[i]

        if angle != 0:
            # Apply rotation at the PDF level via /Rotate attribute.
            # pdf2image already applies the existing /Rotate when rendering,
            # so Tesseract's angle is relative to the currently-displayed view.
            # PDF /Rotate is clockwise; Tesseract's angle is CCW correction needed.
            # Adding the CCW angle as a CW value to /Rotate achieves the correction.
            existing_rotate = int(src_page.get("/Rotate", 0))
            new_rotate = (existing_rotate + angle) % 360
            src_page["/Rotate"] = new_rotate
            print(f"  → Set PDF /Rotate to {new_rotate}° (was {existing_rotate}°)")
        else:
            print(f"  → No rotation needed")

        out_pdf.pages.append(src_page)
        kept += 1
        print()

    if kept == 0:
        print("WARNING: All pages were removed (all blank?). Nothing to save.")
        sys.exit(1)

    print(f"Pages retained: {kept}/{total_pages}")
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
                        help=f"Analysis render DPI for OSD/blank detection (default: {ANALYSIS_DPI})")
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
