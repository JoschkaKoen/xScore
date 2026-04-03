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
        self._log.write(re.sub(r"\x1b\[[0-9;]*m", "", text))
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
    from pipeline.shared.terminal_ui import BOLD, BLUE, icon, note_line, paint

    print()
    print(paint(f"  {icon('spark')}  grade.py  —  Auto-Grader {__version__}", BLUE, BOLD))
    note_line(f"Log file: {log_path}")

    try:
        _run(args, timestamp)
    finally:
        tee.flush()
        tee.close()


def _run(args: argparse.Namespace, timestamp: str) -> None:
    # Late imports after dotenv so env vars are available
    from extraction.providers.kimi import KimiProvider

    from pipeline.shared.terminal_ui import (
        err_line,
        info_line,
        note_line,
        ok_line,
        pipeline_step,
        warn_line,
    )

    from pipeline.marking.detect_answered_questions import detect_answered_exercises
    from pipeline.shared.exam_paths import artifact_scaffold_cache_path, legacy_artifact_scaffold_cache_path
    from pipeline.marking.find_exam_folder import find_folder
    from pipeline.marking.grade_answers import grade_students
    from pipeline.shared.load_ground_truth import evaluate_results, find_ground_truth_file, load_ground_truth
    from pipeline.reports.print_results import (
        print_evaluation_summary,
        print_exercise_summary,
        print_grand_summary,
        print_page_summary,
        print_results_table,
        print_scaffold_summary,
    )
    from pipeline.marking.assign_pages_to_students import assign_pages
    from pipeline.preprocessing.start_scan import cleanup_pdf
    from pipeline.marking.parse_instruction import parse_prompt
    from pipeline.reports.generate_report import generate_report
    from pipeline.scaffold.generate_scaffold import build_scaffold
    from pipeline.shared.load_student_list import read_student_list

    # ------------------------------------------------------------------ #
    # Prerequisite: create shared API client (reused across all steps)   #
    # ------------------------------------------------------------------ #
    client = KimiProvider.create_client()
    if client is None:
        err_line("Could not create Kimi API client.")
        err_line("Set KIMI_API_KEY in your .env file or environment.")
        raise SystemExit(1)

    ok_line("Kimi API client ready.")

    # ------------------------------------------------------------------ #
    # Step 1: Parse natural language prompt                               #
    # ------------------------------------------------------------------ #
    pipeline_step(1, "Parse natural language prompt")
    info_line(f"Prompt: {args.prompt!r}")
    instruction = parse_prompt(args.prompt, client=client, dpi_override=args.dpi)

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

    note_line(
        f"Task: {instruction.task_type}  |  "
        f"Students: {instruction.student_filter.mode}  |  "
        f"DPI: {instruction.dpi}"
    )
    _flag_bits = [
        x
        for x in (
            "skip_clean_scan" if skip_clean_scan else None,
            "force_clean_scan" if force_clean_scan else None,
            "rescaffold" if rescaffold else None,
            f"through_step={through_step}" if through_step is not None else None,
            "no_report" if no_report else None,
        )
        if x
    ]
    if _flag_bits:
        note_line("Effective flags: " + ", ".join(_flag_bits))

    if through_step == 1:
        info_line("through_step 1: stopping after parse prompt (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 2: Find exam folder                                            #
    # ------------------------------------------------------------------ #
    pipeline_step(2, "Find exam folder")
    folder = find_folder(
        instruction_hint=instruction.folder_hint,
        cli_override=args.folder,
        ai_folder_path=None if args.folder else instruction.folder_path,
    )
    note_line(f"{folder}")

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
    note_line(f"Exam output root: {exam_output_root}")
    note_line(f"This run (all artifacts): {artifact_dir}")
    if through_step == 2:
        info_line("through_step 2: stopping after find exam folder (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 3: Read student list                                           #
    # ------------------------------------------------------------------ #
    pipeline_step(3, "Load roster")
    students = read_student_list(folder)
    roster_preview = ", ".join(students[:5]) + (" …" if len(students) > 5 else "")
    note_line(f"{len(students)} students — {roster_preview}")
    if through_step == 3:
        info_line("through_step 3: stopping after load roster (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 4: Build exam scaffold                                         #
    # ------------------------------------------------------------------ #
    pipeline_step(4, "Build exam scaffold")
    if rescaffold:
        for cache_p in (
            artifact_scaffold_cache_path(artifact_dir),
            legacy_artifact_scaffold_cache_path(artifact_dir),
            folder / "scaffolds" / "scaffold_cache.json",
            folder / "scaffold_cache.json",
        ):
            if cache_p.is_file():
                cache_p.unlink()
                warn_line(f"rescaffold: removed {cache_p}")

    scaffold = build_scaffold(folder, client=client, artifact_dir=artifact_dir)
    print_scaffold_summary(scaffold)
    if through_step == 4:
        info_line("through_step 4: stopping after build scaffold (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 5: Clean scan PDF                                              #
    # ------------------------------------------------------------------ #
    pipeline_step(5, "Clean scan PDF")
    if skip_clean_scan:
        cleaned_here = artifact_dir / "cleaned_scan.pdf"
        legacy_cleaned = folder / "cleaned_scan.pdf"
        if cleaned_here.exists():
            cleaned_pdf = cleaned_here
            info_line(f"skip_clean_scan: using {cleaned_pdf}")
        elif legacy_cleaned.exists():
            cleaned_pdf = legacy_cleaned
            info_line(f"skip_clean_scan: using legacy {cleaned_pdf}")
        else:
            scans = list(folder.glob("*.pdf"))
            scans = [f for f in scans if "scan" in f.name.lower()]
            if not scans:
                err_line("skip_clean_scan set but no scan PDF found.")
                raise SystemExit(1)
            cleaned_pdf = scans[0]
            info_line(f"skip_clean_scan: using {cleaned_pdf.name}")
    else:
        cleaned_pdf = cleanup_pdf(
            folder,
            dpi=instruction.dpi,
            force_clean_scan=force_clean_scan,
            artifact_dir=artifact_dir,
        )
    if through_step == 5:
        info_line("through_step 5: stopping after clean scan (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 6: Page assignment                                             #
    # ------------------------------------------------------------------ #
    pipeline_step(6, "Assign pages to students")
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
        proj = cleaned_pdf.with_name(f"{cleaned_pdf.stem}_projected_boxes.pdf")
        info_line("through_step 6: stopping after assign pages (README table).")
        note_line(f"Projected scaffold overlay (if generated): {proj}")
        raise SystemExit(0)

    if not page_map:
        warn_line("No student pages identified. Cannot grade.")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 7: Exercise detection                                          #
    # ------------------------------------------------------------------ #
    pipeline_step(7, "Detect answered exercises")
    exercise_map = detect_answered_exercises(
        cleaned_pdf, page_map, scaffold,
        dpi=NAME_RECOGNITION_DPI, client=client,
    )
    print_exercise_summary(exercise_map)
    if through_step == 7:
        info_line("through_step 7: stopping after exercise detection (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Steps 8–9: Grade and print results                                  #
    # ------------------------------------------------------------------ #
    pipeline_step(8, "Grade submissions")
    results = grade_students(
        cleaned_pdf, page_map, exercise_map, scaffold, instruction, client=client,
    )
    pipeline_step(9, "Print results")
    print_results_table(results, scaffold)
    print_grand_summary(results)
    if through_step in (8, 9):
        info_line(
            f"through_step {through_step}: stopping after grade / results (README table)."
        )
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 10: Ground truth evaluation (if file exists in exam folder)    #
    # ------------------------------------------------------------------ #
    pipeline_step(10, "Ground truth evaluation")
    eval_data: dict | None = None
    gt_file = find_ground_truth_file(folder)
    if gt_file is not None:
        note_line(f"Ground truth file: {gt_file.name}")
        gt = load_ground_truth(folder, scaffold)
        if gt:
            eval_data = evaluate_results(results, gt, scaffold)
            print_evaluation_summary(eval_data, scaffold)
        else:
            warn_line("Ground truth file could not be parsed — skipping evaluation.")
    else:
        info_line("No ground truth file in folder — skipping evaluation.")
        info_line("(Add ground_truth.txt to the exam folder to enable.)")

    if through_step == 10:
        info_line("through_step 10: stopping after ground-truth step (README table).")
        raise SystemExit(0)

    # ------------------------------------------------------------------ #
    # Step 11: PDF report                                                 #
    # ------------------------------------------------------------------ #
    pipeline_step(11, "Generate report")
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
        ok_line(f"Report: {output_pdf}")
    else:
        info_line("no_report: skipping LaTeX/PDF.")
    if through_step == 11:
        ok_line("through_step 11: full pipeline complete (README table).")
        raise SystemExit(0)

    ok_line("Grading pipeline finished.")


if __name__ == "__main__":
    main()
