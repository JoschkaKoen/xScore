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
  4. Build an exam scaffold by parsing vector exam + answer-key PDFs (PyMuPDF).
  5. Clean the scan PDF (auto-rotate + blank removal + deskew); writes under output/<exam_stem>/<run_id>/.
  6. Identify which pages belong to which student.
  7. Detect which exercises each student attempted.
  8. Grade and print a full results table.
  9. Evaluate against ground truth (if a ground_truth.txt file exists in the folder).
 10. Generate a LaTeX/PDF report in the same run directory.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.1"


class _Tee:
    """Duplicate stdout to a log file, stripping ANSI colour codes from the file."""

    def __init__(self, log_path: Path) -> None:
        self._stdout = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self._stdout.write(text)
        # Strip CSI/OSC sequences so log files stay plain (Rich + legacy ANSI).
        plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        plain = re.sub(r"\x1b\][^\x07]*\x07", "", plain)
        self._log.write(plain)
        return len(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._log.flush()

    def isatty(self) -> bool:
        # Delegate so ANSI colors still apply to the real terminal when teeing to a log.
        return self._stdout.isatty()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._log.close()


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
        help="Exam folder path (overrides folder_path / folder_hint from prompt parse)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="N",
        help="Rendering DPI (overrides dpi from prompt parse; default from parse is 400)",
    )
    parser.add_argument(
        "--skip-clean-scan",
        action="store_true",
        default=False,
        help="Skip class-scan prep (OR with same flag from prompt parse)",
    )
    parser.add_argument(
        "--force-clean-scan",
        action="store_true",
        default=False,
        help="Ignore cleaned_scan cache (OR with same flag from prompt parse)",
    )
    parser.add_argument(
        "--rescaffold",
        action="store_true",
        default=False,
        help="Force scaffold rebuild (OR with same flag from prompt parse)",
    )
    parser.add_argument(
        "--through-step",
        type=int,
        default=None,
        metavar="N",
        choices=list(range(1, 12)),
        help="Exit after pipeline step N (1–11); overrides through_step from prompt if set",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        default=False,
        help="Skip LaTeX/PDF report (OR with same flag from prompt parse)",
    )
    args = parser.parse_args()
    if args.skip_clean_scan and args.force_clean_scan:
        parser.error("--skip-clean-scan and --force-clean-scan cannot be used together.")
    return args


def main() -> None:
    load_dotenv()
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path("logs") / f"{timestamp}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    from rich.rule import Rule

    from shared.terminal_ui import get_console, icon, note_line

    c = get_console()
    c.print()
    c.print(
        Rule(
            f"[bold blue]{icon('spark')}  grade.py  —  Auto-Grader {__version__}[/]",
            style="blue",
        )
    )
    try:
        _run(args, timestamp)
    finally:
        tee.flush()
        tee.close()


def _run(args: argparse.Namespace, timestamp: str) -> None:
    # Late imports after dotenv so env vars are available
    from extraction.providers.kimi import KimiProvider

    from shared.terminal_ui import (
        err_line,
        format_duration,
        info_line,
        note_line,
        ok_line,
        pipeline_step,
        warn_line,
    )

    from marking.detect_answered_questions import detect_answered_exercises
    from shared.exam_paths import artifact_scaffold_cache_path, legacy_artifact_scaffold_cache_path
    from marking.find_exam_folder import find_folder
    from marking.grade_answers import grade_students
    from shared.load_ground_truth import evaluate_results, find_ground_truth_file, load_ground_truth
    from reports.print_results import (
        print_evaluation_summary,
        print_exercise_summary,
        print_grand_summary,
        print_page_summary,
        print_results_table,
        print_scaffold_summary,
    )
    from marking.assign_pages_to_students import assign_pages
    from preprocessing.start_scan import cleanup_pdf
    from marking.parse_instruction import parse_prompt
    from reports.generate_report import generate_report
    from scaffold.generate_scaffold import build_scaffold
    from shared.load_student_list import read_student_list

    # ------------------------------------------------------------------ #
    # Prerequisite: create shared API client (reused across all steps)   #
    # ------------------------------------------------------------------ #
    client = KimiProvider.create_client()
    if client is None:
        err_line("Could not create Kimi API client.")
        err_line("Set KIMI_API_KEY in your .env file or environment.")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # Step 1: Parse natural language prompt                               #
    # ------------------------------------------------------------------ #
    pipeline_step(1, "Your request")
    info_line("Parsing …")
    _t_parse = time.perf_counter()
    instruction = parse_prompt(args.prompt, client=client, dpi_override=args.dpi)
    _parse_elapsed = time.perf_counter() - _t_parse

    skip_clean_scan = args.skip_clean_scan or instruction.skip_clean_scan
    force_clean_scan = args.force_clean_scan or instruction.force_clean_scan
    if skip_clean_scan and force_clean_scan:
        err_line("Cannot combine skip and force class-scan cleaning (CLI and/or prompt).")
        raise SystemExit(1)
    rescaffold = args.rescaffold or instruction.rescaffold
    through_step = (
        args.through_step
        if args.through_step is not None
        else instruction.through_step
    )
    no_report = args.no_report or instruction.no_report

    _task_labels = {
        "check_answers": "Grade answers",
        "check_mc": "Multiple choice only",
        "count_marks": "Count marks",
        "build_scaffold": "Build structure",
        "clean_scan": "Clean scan",
    }
    _task_label = _task_labels.get(
        instruction.task_type,
        instruction.task_type.replace("_", " ").strip(),
    )
    _sf = instruction.student_filter
    if _sf.mode == "all":
        _scope = "all students"
    elif _sf.mode == "first_n" and _sf.n > 0:
        _scope = f"first {_sf.n} students"
    elif _sf.names:
        _scope = f"{len(_sf.names)} named students"
    else:
        _scope = _sf.mode.replace("_", " ")
    ok_line(
        f"{_task_label}  ·  {_scope}  ·  {instruction.dpi} DPI  ·  "
        f"{format_duration(_parse_elapsed)}"
    )

    if through_step == 1:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 2: Find exam folder                                            #
    # ------------------------------------------------------------------ #
    pipeline_step(2, "Exam folder")
    folder = find_folder(
        instruction_hint=instruction.folder_hint,
        cli_override=args.folder,
        ai_folder_path=None if args.folder else instruction.folder_path,
    )

    stem = folder.name.replace(" ", "_")
    exam_output_root = Path("output") / stem
    exam_output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = exam_output_root / timestamp
    suffix = 1
    while artifact_dir.exists():
        suffix += 1
        artifact_dir = exam_output_root / f"{timestamp}_{suffix}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Reports and all derived files for this invocation live in the same run folder.
    run_dir = artifact_dir
    ok_line(folder.name)
    if through_step == 2:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 3: Read student list                                           #
    # ------------------------------------------------------------------ #
    pipeline_step(3, "Students")
    students = read_student_list(folder)
    ok_line(f"{len(students)} students on the roster")
    if through_step == 3:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 4: Build exam scaffold                                         #
    # ------------------------------------------------------------------ #
    pipeline_step(4, "Mark scheme")
    if rescaffold:
        for cache_p in (
            artifact_scaffold_cache_path(artifact_dir),
            legacy_artifact_scaffold_cache_path(artifact_dir),
            folder / "scaffolds" / "scaffold_cache.json",
            folder / "scaffold_cache.json",
        ):
            if cache_p.is_file():
                cache_p.unlink()
                warn_line("Removed cached scaffold (rebuild).")

    scaffold = build_scaffold(folder, client=client, artifact_dir=artifact_dir)
    print_scaffold_summary(scaffold)
    if through_step == 4:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 5: Clean scan PDF                                              #
    # ------------------------------------------------------------------ #
    pipeline_step(5, "Scan")
    if skip_clean_scan:
        cleaned_here = artifact_dir / "cleaned_scan.pdf"
        legacy_cleaned = folder / "cleaned_scan.pdf"
        if cleaned_here.exists():
            cleaned_pdf = cleaned_here
            info_line("Using existing cleaned scan (skip).")
        elif legacy_cleaned.exists():
            cleaned_pdf = legacy_cleaned
            info_line("Using existing cleaned scan (skip).")
        else:
            scans = list(folder.glob("*.pdf"))
            scans = [f for f in scans if "scan" in f.name.lower()]
            if not scans:
                err_line("skip_clean_scan set but no scan PDF found.")
                raise SystemExit(1)
            cleaned_pdf = scans[0]
            info_line("Using existing scan PDF (skip).")
    else:
        cleaned_pdf = cleanup_pdf(
            folder,
            dpi=instruction.dpi,
            force_clean_scan=force_clean_scan,
            artifact_dir=artifact_dir,
        )
    if through_step == 5:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 6: Page assignment                                             #
    # ------------------------------------------------------------------ #
    pipeline_step(6, "Page assignment")
    from config import NAME_CROP_FRACTION, NAME_RECOGNITION_DPI

    page_map = assign_pages(
        cleaned_pdf,
        students,
        dpi=NAME_RECOGNITION_DPI,
        client=client,
        name_crop_fraction=NAME_CROP_FRACTION,
        verbose=False,
    )
    print_page_summary(page_map, students)
    if through_step == 6:
        raise SystemExit(0)

    if not page_map:
        warn_line("No student pages identified. Cannot grade.")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 7: Exercise detection                                          #
    # ------------------------------------------------------------------ #
    pipeline_step(7, "Questions attempted")
    exercise_map = detect_answered_exercises(
        cleaned_pdf, page_map, scaffold,
        dpi=NAME_RECOGNITION_DPI, client=client,
    )
    print_exercise_summary(exercise_map)
    if through_step == 7:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Steps 8–9: Grade and print results                                  #
    # ------------------------------------------------------------------ #
    pipeline_step(8, "Marking")
    results = grade_students(
        cleaned_pdf, page_map, exercise_map, scaffold, instruction, client=client,
    )
    pipeline_step(9, "Results")
    print_results_table(results, scaffold)
    print_grand_summary(results)
    if through_step in (8, 9):
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 10: Ground truth evaluation (if file exists in exam folder)    #
    # ------------------------------------------------------------------ #
    pipeline_step(10, "Accuracy check")
    eval_data: dict | None = None
    gt_file = find_ground_truth_file(folder)
    if gt_file is not None:
        info_line("Reference list found — comparing to extracted answers.")
        gt = load_ground_truth(folder, scaffold)
        if gt:
            eval_data = evaluate_results(results, gt, scaffold)
            print_evaluation_summary(eval_data, scaffold)
        else:
            warn_line("Ground truth file could not be parsed — skipping evaluation.")
    else:
        info_line("No reference list in the exam folder — skipped.")

    if through_step == 10:
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 11: PDF report                                                 #
    # ------------------------------------------------------------------ #
    pipeline_step(11, "Report")
    if not no_report:
        output_tex = run_dir / "grade_report.tex"
        output_pdf = run_dir / "grade_report.pdf"
        title = f"{folder.name} — Grading Report"
        generate_report(
            scaffold=scaffold,
            results=results,
            output_tex=output_tex,
            output_pdf=output_pdf,
            eval_data=eval_data,
            title=title,
        )
        ok_line("Report saved.")
    else:
        info_line("PDF report skipped (you turned it off).")
    if through_step == 11:
        ok_line("Pipeline complete.")
        raise SystemExit(0)

    ok_line("Grading pipeline finished.")


if __name__ == "__main__":
    main()
