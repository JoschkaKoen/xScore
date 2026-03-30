#!/usr/bin/env python3
"""
fix_scanned_pdf.py
------------------
Cleans up scanned exam PDFs by:
  1. Auto-rotating pages to upright orientation (via Tesseract OSD)
  2. Removing blank/white pages
  3. Saving the result as a new PDF

Requirements:
    pip install pdf2image pytesseract numpy img2pdf Pillow
    apt install tesseract-ocr poppler-utils   # or brew install on macOS


    cd /Users/joschka/Desktop/Programming/Auto-Grader
    source .venv/bin/activate
    python autograder.py input.pdf output.pdf

Usage:
    python fix_scanned_pdf.py input.pdf output.pdf
    python fix_scanned_pdf.py input.pdf output.pdf --dpi 200 --blank-threshold 250 --blank-std 5
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import img2pdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_DPI = 150           # Render resolution. 150 is fast; use 200-300 for quality.
BLANK_MEAN_THRESHOLD = 250  # Pages with grayscale mean above this are considered blank (0-255)
BLANK_STD_THRESHOLD = 5     # Pages with grayscale std below this are considered blank


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_rotation(image: Image.Image) -> int:
    """
    Use Tesseract OSD to detect how many degrees CCW the page needs to be
    rotated to appear upright.

    Returns one of: 0, 90, 180, 270
    Returns 0 if detection fails (e.g. nearly blank page with no text).
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))

        # Low confidence means Tesseract is guessing — don't rotate
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


def pil_to_pdf_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image to JPEG bytes suitable for img2pdf."""
    buf = io.BytesIO()
    # Convert to RGB — img2pdf doesn't accept RGBA or palette modes
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def process_pdf(input_path: str,
                output_path: str,
                dpi: int = DEFAULT_DPI,
                blank_mean: float = BLANK_MEAN_THRESHOLD,
                blank_std: float = BLANK_STD_THRESHOLD) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"\nRendering pages from: {input_path}")
    print(f"DPI: {dpi}  |  Blank thresholds: mean≥{blank_mean}, std≤{blank_std}\n")

    pages = convert_from_path(str(input_path), dpi=dpi)
    print(f"Total pages in input: {len(pages)}\n")

    processed_images = []

    for i, page_img in enumerate(pages):
        page_num = i + 1
        print(f"Page {page_num}/{len(pages)}")

        # --- Blank page check ---
        if is_blank_page(page_img, blank_mean, blank_std):
            print(f"  → Removing blank page\n")
            continue

        # --- Orientation correction ---
        angle = detect_rotation(page_img)
        if angle != 0:
            # PIL rotate: positive = CCW, expand=True adjusts canvas for 90/270
            page_img = page_img.rotate(angle, expand=True)
            print(f"  → Rotated {angle}° CCW")
        else:
            print(f"  → No rotation needed")

        processed_images.append(page_img)
        print()

    if not processed_images:
        print("WARNING: All pages were removed (all blank?). Nothing to save.")
        sys.exit(1)

    print(f"Pages retained: {len(processed_images)}/{len(pages)}")
    print(f"Saving to: {output_path}")

    # Convert PIL images → JPEG bytes → single PDF via img2pdf
    jpeg_pages = [pil_to_pdf_bytes(img) for img in processed_images]
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(jpeg_pages))

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
    parser.add_argument("--dpi",  type=int,   default=DEFAULT_DPI,
                        help=f"Render DPI (default: {DEFAULT_DPI})")
    parser.add_argument("--blank-threshold", type=float, default=BLANK_MEAN_THRESHOLD,
                        help=f"Grayscale mean above which a page is blank (default: {BLANK_MEAN_THRESHOLD})")
    parser.add_argument("--blank-std", type=float, default=BLANK_STD_THRESHOLD,
                        help=f"Grayscale std below which a page is blank (default: {BLANK_STD_THRESHOLD})")
    args = parser.parse_args()

    process_pdf(
        input_path=args.input,
        output_path=args.output,
        dpi=args.dpi,
        blank_mean=args.blank_threshold,
        blank_std=args.blank_std,
    )


if __name__ == "__main__":
    main()