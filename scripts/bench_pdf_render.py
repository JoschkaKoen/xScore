#!/usr/bin/env python3
"""
Time pdf2image's convert_from_path only (no Gemini, no extraction).
Usage:
    .venv/bin/python scripts/bench_pdf_render.py [path/to.pdf] [--dpi 300]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DEFAULT_PDF
from pdf2image import convert_from_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PDF→raster at given DPI.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default=DEFAULT_PDF,
        help="Input PDF path",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"ERROR: PDF not found: {pdf_path}")

    cores = os.cpu_count() or 1
    print(f"Rendering: {pdf_path.resolve()}")
    print(f"DPI: {args.dpi}  |  threads: {cores} (os.cpu_count)\n")

    t0 = time.perf_counter()
    pages = convert_from_path(str(pdf_path), dpi=args.dpi, thread_count=cores)
    elapsed = time.perf_counter() - t0

    n = len(pages)
    per = elapsed / n if n else 0.0
    print(f"PDF→images ({args.dpi} DPI, {cores} threads): {elapsed:.2f}s total ({per:.2f}s/page) — {n} pages")


if __name__ == "__main__":
    main()
