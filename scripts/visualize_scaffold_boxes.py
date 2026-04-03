#!/usr/bin/env python3
"""Draw red rectangles for scaffold bounding boxes on a copy of the vector exam PDF.

Loads scaffold cache from ``output/<stem>/scaffolds/`` (or legacy paths under the exam
folder). Default overlay PDF: ``output/<stem>/overlays/<exam_stem>_scaffold_boxes.pdf``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.exam_paths import (
    artifact_overlays_dir,
    exam_artifact_dir,
    find_scaffold_cache_file,
    safe_path_stem,
)
from pipeline.scaffold import _find_exam_pdf, question_from_dict
from pipeline.scaffold_overlay import write_scaffold_boxes_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Exam folder (raw exam PDF); scaffold cache in output/<stem>/ or legacy paths",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: output/<stem>/overlays/..._scaffold_boxes.pdf)",
    )
    args = ap.parse_args()
    folder = args.folder.resolve()
    cache = find_scaffold_cache_file(folder)
    if cache is None:
        ad = exam_artifact_dir(folder)
        print(
            f"Missing scaffold cache (looked under {ad}/scaffolds/ and exam folder).",
            file=sys.stderr,
        )
        sys.exit(1)

    exam_pdf = _find_exam_pdf(folder)
    with open(cache, encoding="utf-8") as f:
        data = json.load(f)
    roots = [question_from_dict(q) for q in data["questions"]]

    default_out = artifact_overlays_dir(exam_artifact_dir(folder)) / (
        f"{safe_path_stem(exam_pdf.stem)}_scaffold_boxes.pdf"
    )
    out_pdf, n_rects, n_pages = write_scaffold_boxes_pdf(
        exam_pdf, roots, args.output if args.output is not None else default_out
    )
    print(f"Wrote {out_pdf} ({n_rects} rectangles on {n_pages} page(s)).")


if __name__ == "__main__":
    main()
