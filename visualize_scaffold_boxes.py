#!/usr/bin/env python3
"""Draw red rectangles for scaffold bounding boxes on a copy of the vector exam PDF.

Loads ``{folder}/scaffolds/scaffold_cache.json``. For automatic output after a fresh
parse, see :func:`pipeline.scaffold.build_scaffold` (writes the same file next to the
raw exam PDF).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.scaffold import _cache_path, _find_exam_pdf, question_from_dict
from pipeline.scaffold_overlay import write_scaffold_boxes_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Exam folder containing raw exam PDF and scaffolds/scaffold_cache.json",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: next to raw exam, stem + _scaffold_boxes.pdf)",
    )
    args = ap.parse_args()
    folder = args.folder.resolve()
    cache = _cache_path(folder)
    if not cache.is_file():
        print(f"Missing scaffold cache: {cache}", file=sys.stderr)
        sys.exit(1)

    exam_pdf = _find_exam_pdf(folder)
    with open(cache, encoding="utf-8") as f:
        data = json.load(f)
    roots = [question_from_dict(q) for q in data["questions"]]

    out_pdf, n_rects, n_pages = write_scaffold_boxes_pdf(
        exam_pdf, roots, args.output
    )
    print(f"Wrote {out_pdf} ({n_rects} rectangles on {n_pages} page(s)).")


if __name__ == "__main__":
    main()
