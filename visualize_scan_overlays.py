#!/usr/bin/env python3
"""Write deskew debug PDFs next to a cleaned/deskewed scan.

Produces (when inputs allow):

- ``{stem}_reflines_overlay.pdf`` — vertical reference lines + IGCSE anchor marks
  from ``{stem}_reflines.json`` (from :func:`pipeline.scan_deskew.deskew_pdf_raster`).
- ``{stem}_projected_boxes.pdf`` — scaffold regions projected from 4-up PDF space
  onto the raster scan (see :func:`pipeline.bbox_projection.overlay_projected_scaffold_on_scan_pdf`).

The vector exam overlay (``*_scaffold_boxes.pdf``) is produced by
:func:`pipeline.scaffold.build_scaffold` / ``visualize_scaffold_boxes.py`` — not here.

Examples::

    python visualize_scan_overlays.py "Space Physics Unit Test"
    python visualize_scan_overlays.py "Space Physics Unit Test" \\
        --scan "Space Physics Unit Test/cleaned_trial_300dpi.pdf" \\
        --force-projected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.scan_overlays import (
    write_projected_scaffold_debug_pdf,
    write_reflines_debug_pdf,
    write_scan_debug_pdfs_after_deskew,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Exam folder (scaffold cache + optional raw exam 4up)",
    )
    ap.add_argument(
        "--scan",
        type=Path,
        default=None,
        help="Deskewed raster PDF (default: <folder>/cleaned_scan.pdf)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterisation DPI of the scan PDF (default: 300)",
    )
    ap.add_argument(
        "--reflines-only",
        action="store_true",
        help="Only write the reflines overlay PDF",
    )
    ap.add_argument(
        "--projected-only",
        action="store_true",
        help="Only write the projected scaffold overlay PDF",
    )
    ap.add_argument(
        "--force-projected",
        action="store_true",
        help="Draw projected boxes even if scaffold exam PDF is not the 4-up file",
    )
    args = ap.parse_args()

    folder = args.folder.resolve()
    scan = args.scan.resolve() if args.scan else folder / "cleaned_scan.pdf"
    if not scan.is_file():
        print(f"Missing scan PDF: {scan}", file=sys.stderr)
        sys.exit(1)

    rfl = args.reflines_only and not args.projected_only
    prj = args.projected_only and not args.reflines_only
    if rfl:
        write_reflines_debug_pdf(scan, args.dpi)
    elif prj:
        write_projected_scaffold_debug_pdf(
            folder, scan, args.dpi, force_layout_mismatch=args.force_projected
        )
    else:
        write_scan_debug_pdfs_after_deskew(
            folder,
            scan,
            args.dpi,
            force_projected_mismatch=args.force_projected,
        )


if __name__ == "__main__":
    main()
