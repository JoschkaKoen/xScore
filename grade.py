#!/usr/bin/env python3
"""
grade.py
--------
Generic exam grading CLI — works on any exam, driven by a natural language prompt.

Usage:
    python grade.py "check all multiple choice question answers"
    python grade.py "count marks for each student" --folder "Space Physics Unit Test"
    python grade.py "check the first 5 students' answers" --dpi 300
    python grade.py "check answers for Alice and Bob" --folder "Maths Mock"

The program will:
  1. Parse the natural language prompt into structured instructions (via Kimi).
  2. Locate the exam folder.
  3. Read the student roster from StudentList.xlsx.
  4. Build an exam scaffold by AI-analysing the raw exam + answer key PDFs.
  5. Clean the scan PDF (auto-rotate + blank removal).
  6. Identify which pages belong to which student.
  7. Detect which exercises each student attempted.
  8. Grade and print a full results table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from version import __version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="grade.py",
        description="Grade any exam from a natural language prompt.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "prompt",
        help='Grading instruction, e.g. "check all multiple choice question answers"',
    )
    parser.add_argument(
        "--folder",
        default=None,
        metavar="PATH",
        help="Exam folder path (overrides AI-detected folder hint)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="N",
        help="Rendering DPI for page images (overrides AI-detected DPI, default 400)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Skip PDF cleaning step (use existing cleaned_scan.pdf if present)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Late imports after dotenv so env vars are available
    from extraction.providers.kimi import KimiProvider

    from pipeline.answer_detection import detect_answered_exercises
    from pipeline.folder_discovery import find_folder
    from pipeline.grading import grade_students
    from pipeline.output import (
        print_exercise_summary,
        print_grand_summary,
        print_page_summary,
        print_results_table,
        print_scaffold_summary,
    )
    from pipeline.page_assignment import assign_pages
    from pipeline.pdf_cleanup import cleanup_pdf
    from pipeline.prompt_parser import parse_prompt
    from pipeline.scaffold import build_scaffold
    from pipeline.student_list import read_student_list

    # ------------------------------------------------------------------ #
    # Step 1: Create shared API client (reused across all pipeline steps) #
    # ------------------------------------------------------------------ #
    client = KimiProvider.create_client()
    if client is None:
        print(
            "ERROR: Could not create Kimi API client.\n"
            "Set KIMI_API_KEY in your .env file or environment.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # Step 2: Parse natural language prompt                               #
    # ------------------------------------------------------------------ #
    print(f"\n[grade] Parsing prompt: {args.prompt!r}")
    instruction = parse_prompt(args.prompt, client=client, dpi_override=args.dpi)
    print(
        f"[grade] Task: {instruction.task_type}  |  "
        f"Students: {instruction.student_filter.mode}  |  "
        f"DPI: {instruction.dpi}"
    )

    # ------------------------------------------------------------------ #
    # Step 3: Find exam folder                                            #
    # ------------------------------------------------------------------ #
    folder = find_folder(
        instruction_hint=instruction.folder_hint,
        cli_override=args.folder,
    )
    print(f"[grade] Exam folder: {folder}")

    # ------------------------------------------------------------------ #
    # Step 4: Read student list                                           #
    # ------------------------------------------------------------------ #
    students = read_student_list(folder)
    print(f"[grade] Roster: {len(students)} students — {', '.join(students[:5])}" +
          (" …" if len(students) > 5 else ""))

    # ------------------------------------------------------------------ #
    # Step 5: Build exam scaffold                                         #
    # ------------------------------------------------------------------ #
    scaffold = build_scaffold(folder, client=client)
    print_scaffold_summary(scaffold)

    # ------------------------------------------------------------------ #
    # Step 6: Clean scan PDF                                              #
    # ------------------------------------------------------------------ #
    if args.no_cleanup:
        # Use pre-existing cleaned_scan.pdf or raw scan
        existing = folder / "cleaned_scan.pdf"
        if existing.exists():
            cleaned_pdf = existing
            print(f"[grade] --no-cleanup: using {cleaned_pdf}")
        else:
            scans = list(folder.glob("*.pdf"))
            scans = [f for f in scans if "scan" in f.name.lower()]
            if not scans:
                print("ERROR: --no-cleanup set but no scan PDF found.", file=sys.stderr)
                raise SystemExit(1)
            cleaned_pdf = scans[0]
            print(f"[grade] --no-cleanup: using {cleaned_pdf}")
    else:
        cleaned_pdf = cleanup_pdf(folder, dpi=instruction.dpi)

    # ------------------------------------------------------------------ #
    # Step 7: Page assignment                                             #
    # ------------------------------------------------------------------ #
    from config import NAME_CROP_FRACTION, NAME_RECOGNITION_DPI

    page_map = assign_pages(
        cleaned_pdf,
        students,
        dpi=NAME_RECOGNITION_DPI,
        client=client,
        name_crop_fraction=NAME_CROP_FRACTION,
    )
    print_page_summary(page_map, students)

    if not page_map:
        print("WARNING: No student pages identified. Cannot grade.", file=sys.stderr)
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 8: Exercise detection                                          #
    # ------------------------------------------------------------------ #
    exercise_map = detect_answered_exercises(
        cleaned_pdf, page_map, scaffold,
        dpi=NAME_RECOGNITION_DPI, client=client,
    )
    print_exercise_summary(exercise_map)

    # ------------------------------------------------------------------ #
    # Step 9: Grade                                                       #
    # ------------------------------------------------------------------ #
    results = grade_students(
        cleaned_pdf, page_map, exercise_map, scaffold, instruction, client=client,
    )
    print_results_table(results, scaffold)
    print_grand_summary(results)


if __name__ == "__main__":
    main()
