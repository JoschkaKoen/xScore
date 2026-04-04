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
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from shared.models import ExamScaffold, PageAssignment, StudentResult, TaskInstruction

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


@dataclass
class _GradeCtx:
    """Mutable state passed through private pipeline steps (``grade.py``)."""

    args: argparse.Namespace
    timestamp: str
    client: Any = None
    instruction: TaskInstruction | None = None
    parse_elapsed: float = 0.0
    skip_clean_scan: bool = False
    force_clean_scan: bool = False
    rescaffold: bool = False
    through_step: int | None = None
    no_report: bool = False
    folder: Path | None = None
    run_dir: Path | None = None
    artifact_dir: Path | None = None
    students: list[str] | None = None
    scaffold: ExamScaffold | None = None
    cleaned_pdf: Path | None = None
    name_pages: list | None = None
    thread_count: int = 4
    page_map: list[PageAssignment] | None = None
    grade_render_ex: Any = None
    grade_pages_future: Any = None
    exercise_map: dict[str, list[str]] | None = None
    grade_pages: list | None = None
    results: list[StudentResult] | None = None
    eval_data: dict | None = None
    # Set only before SystemExit(0) from --through-step / prompt through_step
    partial_stop_readme_step: int | None = None
    # True only after full success (no through-step early exit)
    pipeline_completed_ok: bool = False


def _print_grade_run_footer(ctx: _GradeCtx, gi: SimpleNamespace, elapsed: float) -> None:
    """Always print wall time; note partial or full pipeline in one line when useful."""
    t = f"{elapsed:.1f}s"
    if ctx.partial_stop_readme_step is not None:
        n = ctx.partial_stop_readme_step
        gi.info_line(f"Run · {t} · partial {n}/11")
    elif ctx.pipeline_completed_ok:
        gi.info_line(f"Run · {t} · complete")
    else:
        gi.info_line(f"Run · {t}")


def _load_grade_imports() -> SimpleNamespace:
    """Late imports after ``load_dotenv()`` so environment is available."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from extraction.providers.kimi import KimiProvider
    from pdf2image import convert_from_path

    from config import NAME_CROP_FRACTION, NAME_RECOGNITION_DPI, pipeline_ai_model_display_name
    from marking.assign_pages_to_students import assign_pages
    from marking.detect_answered_questions import detect_answered_exercises
    from marking.find_exam_folder import find_folder
    from marking.grade_answers import grade_students
    from marking.parse_instruction import parse_prompt
    from preprocessing.start_scan import cleanup_pdf
    from reports.generate_report import generate_report
    from reports.print_results import (
        print_evaluation_summary,
        print_exercise_summary,
        print_grand_summary,
        print_page_summary,
        print_results_table,
        print_scaffold_summary,
    )
    from scaffold.generate_scaffold import build_scaffold
    from shared.exam_paths import artifact_scaffold_cache_path, legacy_artifact_scaffold_cache_path
    from shared.load_ground_truth import evaluate_results, find_ground_truth_file, load_ground_truth
    from shared.load_student_list import read_student_list
    from shared.terminal_ui import (
        err_line,
        format_duration,
        get_console,
        info_line,
        ok_line,
        pipeline_step,
        warn_line,
    )

    return SimpleNamespace(
        ThreadPoolExecutor=ThreadPoolExecutor,
        partial=partial,
        convert_from_path=convert_from_path,
        KimiProvider=KimiProvider,
        NAME_CROP_FRACTION=NAME_CROP_FRACTION,
        NAME_RECOGNITION_DPI=NAME_RECOGNITION_DPI,
        pipeline_ai_model_display_name=pipeline_ai_model_display_name,
        assign_pages=assign_pages,
        detect_answered_exercises=detect_answered_exercises,
        find_folder=find_folder,
        grade_students=grade_students,
        parse_prompt=parse_prompt,
        cleanup_pdf=cleanup_pdf,
        generate_report=generate_report,
        print_evaluation_summary=print_evaluation_summary,
        print_exercise_summary=print_exercise_summary,
        print_grand_summary=print_grand_summary,
        print_page_summary=print_page_summary,
        print_results_table=print_results_table,
        print_scaffold_summary=print_scaffold_summary,
        build_scaffold=build_scaffold,
        artifact_scaffold_cache_path=artifact_scaffold_cache_path,
        legacy_artifact_scaffold_cache_path=legacy_artifact_scaffold_cache_path,
        evaluate_results=evaluate_results,
        find_ground_truth_file=find_ground_truth_file,
        load_ground_truth=load_ground_truth,
        read_student_list=read_student_list,
        err_line=err_line,
        format_duration=format_duration,
        get_console=get_console,
        info_line=info_line,
        ok_line=ok_line,
        pipeline_step=pipeline_step,
        warn_line=warn_line,
    )


def _grade_create_client(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    ctx.client = gi.KimiProvider.create_client()
    if ctx.client is None:
        gi.err_line("Could not create Kimi API client.")
        gi.err_line("Set KIMI_API_KEY in your .env file or environment.")
        raise SystemExit(1)


def _grade_step01_parse(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    gi.pipeline_step(1, "Your request")
    gi.info_line(f"Parsing prompt with {gi.pipeline_ai_model_display_name()} …")
    t0 = time.perf_counter()
    ctx.instruction = gi.parse_prompt(ctx.args.prompt, client=ctx.client, dpi_override=ctx.args.dpi)
    ctx.parse_elapsed = time.perf_counter() - t0
    assert ctx.instruction is not None
    inst = ctx.instruction

    ctx.skip_clean_scan = ctx.args.skip_clean_scan or inst.skip_clean_scan
    ctx.force_clean_scan = ctx.args.force_clean_scan or inst.force_clean_scan
    if ctx.skip_clean_scan and ctx.force_clean_scan:
        gi.err_line("Cannot combine skip and force class-scan cleaning (CLI and/or prompt).")
        raise SystemExit(1)
    ctx.rescaffold = ctx.args.rescaffold or inst.rescaffold
    ctx.through_step = (
        ctx.args.through_step if ctx.args.through_step is not None else inst.through_step
    )
    ctx.no_report = ctx.args.no_report or inst.no_report

    task_labels = {
        "check_answers": "Grade answers",
        "check_mc": "Multiple choice only",
        "count_marks": "Count marks",
        "build_scaffold": "Build structure",
        "clean_scan": "Clean scan",
    }
    task_label = task_labels.get(inst.task_type, inst.task_type.replace("_", " ").strip())
    sf = inst.student_filter
    if sf.mode == "all":
        scope = "all students"
    elif sf.mode == "first_n" and sf.n > 0:
        scope = f"first {sf.n} students"
    elif sf.names:
        scope = f"{len(sf.names)} named students"
    else:
        scope = sf.mode.replace("_", " ")
    gi.ok_line(
        f"{task_label}  ·  {scope}  ·  {inst.dpi} DPI  ·  "
        f"{gi.format_duration(ctx.parse_elapsed)}"
    )

    if ctx.through_step == 1:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step02_folder(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert ctx.instruction is not None
    gi.pipeline_step(2, "Exam folder")
    ctx.folder = gi.find_folder(
        instruction_hint=ctx.instruction.folder_hint,
        cli_override=ctx.args.folder,
        ai_folder_path=None if ctx.args.folder else ctx.instruction.folder_path,
    )
    assert ctx.folder is not None
    stem = ctx.folder.name.replace(" ", "_")
    exam_output_root = Path("output") / stem
    exam_output_root.mkdir(parents=True, exist_ok=True)
    ctx.artifact_dir = exam_output_root / ctx.timestamp
    suffix = 1
    while ctx.artifact_dir.exists():
        suffix += 1
        ctx.artifact_dir = exam_output_root / f"{ctx.timestamp}_{suffix}"
    ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
    ctx.run_dir = ctx.artifact_dir
    gi.ok_line(ctx.folder.name)
    if ctx.through_step == 2:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step03_students(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert ctx.folder is not None
    gi.pipeline_step(3, "Students")
    ctx.students = gi.read_student_list(ctx.folder)
    gi.ok_line(f"{len(ctx.students)} students on the roster")
    if ctx.through_step == 3:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step04_scaffold(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert ctx.folder is not None and ctx.artifact_dir is not None and ctx.client is not None
    gi.pipeline_step(4, "Mark scheme")
    if ctx.rescaffold:
        for cache_p in (
            gi.artifact_scaffold_cache_path(ctx.artifact_dir),
            gi.legacy_artifact_scaffold_cache_path(ctx.artifact_dir),
            ctx.folder / "scaffolds" / "scaffold_cache.json",
            ctx.folder / "scaffold_cache.json",
        ):
            if cache_p.is_file():
                cache_p.unlink()
                gi.warn_line("Removed cached scaffold (rebuild).")

    ctx.scaffold = gi.build_scaffold(ctx.folder, client=ctx.client, artifact_dir=ctx.artifact_dir)
    gi.print_scaffold_summary(ctx.scaffold)
    if ctx.through_step == 4:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step05_clean_scan(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert ctx.folder is not None and ctx.artifact_dir is not None and ctx.instruction is not None
    gi.pipeline_step(5, "Scan")
    gi.get_console().print()
    if ctx.skip_clean_scan:
        cleaned_here = ctx.artifact_dir / "cleaned_scan.pdf"
        legacy_cleaned = ctx.folder / "cleaned_scan.pdf"
        if cleaned_here.exists():
            ctx.cleaned_pdf = cleaned_here
            gi.info_line("Using existing cleaned scan (skip).")
        elif legacy_cleaned.exists():
            ctx.cleaned_pdf = legacy_cleaned
            gi.info_line("Using existing cleaned scan (skip).")
        else:
            scans = list(ctx.folder.glob("*.pdf"))
            scans = [f for f in scans if "scan" in f.name.lower()]
            if not scans:
                gi.err_line("skip_clean_scan set but no scan PDF found.")
                raise SystemExit(1)
            ctx.cleaned_pdf = scans[0]
            gi.info_line("Using existing scan PDF (skip).")
    else:
        ctx.cleaned_pdf = gi.cleanup_pdf(
            ctx.folder,
            dpi=ctx.instruction.dpi,
            force_clean_scan=ctx.force_clean_scan,
            artifact_dir=ctx.artifact_dir,
        )
    if ctx.through_step == 5:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step06_page_assignment(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert (
        ctx.cleaned_pdf is not None
        and ctx.students is not None
        and ctx.instruction is not None
        and ctx.client is not None
    )
    gi.pipeline_step(6, "Page assignment")
    ctx.thread_count = os.cpu_count() or 4
    gi.info_line(
        f"Rendering pages for name + exercise detection @ {gi.NAME_RECOGNITION_DPI} DPI …"
    )
    t0 = time.perf_counter()
    ctx.name_pages = gi.convert_from_path(
        str(ctx.cleaned_pdf),
        dpi=gi.NAME_RECOGNITION_DPI,
        thread_count=ctx.thread_count,
    )
    gi.ok_line(f"Pages loaded · {gi.format_duration(time.perf_counter() - t0)}")

    ctx.page_map = gi.assign_pages(
        ctx.cleaned_pdf,
        ctx.students,
        dpi=gi.NAME_RECOGNITION_DPI,
        client=ctx.client,
        name_crop_fraction=gi.NAME_CROP_FRACTION,
        verbose=False,
        pages=ctx.name_pages,
    )
    gi.print_page_summary(ctx.page_map, ctx.students)
    if ctx.through_step == 6:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)

    if not ctx.page_map:
        gi.warn_line("No student pages identified. Cannot grade.")
        raise SystemExit(0)

    will_grade = ctx.through_step is None or (
        isinstance(ctx.through_step, int) and ctx.through_step >= 8
    )
    if will_grade:
        ctx.grade_render_ex = gi.ThreadPoolExecutor(max_workers=1)
        ctx.grade_pages_future = ctx.grade_render_ex.submit(
            gi.partial(
                gi.convert_from_path,
                str(ctx.cleaned_pdf),
                dpi=ctx.instruction.dpi,
                thread_count=ctx.thread_count,
            ),
        )


def _grade_step07_detect(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert (
        ctx.cleaned_pdf is not None
        and ctx.page_map is not None
        and ctx.scaffold is not None
        and ctx.name_pages is not None
        and ctx.client is not None
    )
    gi.pipeline_step(7, "Questions attempted")
    ctx.exercise_map = gi.detect_answered_exercises(
        ctx.cleaned_pdf,
        ctx.page_map,
        ctx.scaffold,
        dpi=gi.NAME_RECOGNITION_DPI,
        client=ctx.client,
        pages=ctx.name_pages,
    )
    gi.print_exercise_summary(ctx.exercise_map)
    if ctx.through_step == 7:
        if ctx.grade_render_ex is not None:
            ctx.grade_render_ex.shutdown(wait=False, cancel_futures=True)
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step08_09_grade(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert (
        ctx.cleaned_pdf is not None
        and ctx.page_map is not None
        and ctx.exercise_map is not None
        and ctx.scaffold is not None
        and ctx.instruction is not None
        and ctx.client is not None
    )
    gi.pipeline_step(8, "Marking")
    ctx.grade_pages = None
    if ctx.grade_pages_future is not None:
        gi.info_line(f"Awaiting marking render @ {ctx.instruction.dpi} DPI …")
        t0 = time.perf_counter()
        ctx.grade_pages = ctx.grade_pages_future.result()
        gi.ok_line(f"Pages ready · {gi.format_duration(time.perf_counter() - t0)}")
        if ctx.grade_render_ex is not None:
            ctx.grade_render_ex.shutdown(wait=False)
            ctx.grade_render_ex = None

    ctx.results = gi.grade_students(
        ctx.cleaned_pdf,
        ctx.page_map,
        ctx.exercise_map,
        ctx.scaffold,
        ctx.instruction,
        client=ctx.client,
        pages=ctx.grade_pages,
    )
    gi.pipeline_step(9, "Results")
    gi.print_results_table(ctx.results, ctx.scaffold)
    gi.print_grand_summary(ctx.results)
    if ctx.through_step in (8, 9):
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step10_eval(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert ctx.folder is not None and ctx.scaffold is not None and ctx.results is not None
    gi.pipeline_step(10, "Accuracy check")
    ctx.eval_data = None
    gt_file = gi.find_ground_truth_file(ctx.folder)
    if gt_file is not None:
        gi.info_line("Reference list found — comparing to extracted answers.")
        gt = gi.load_ground_truth(ctx.folder, ctx.scaffold)
        if gt:
            ctx.eval_data = gi.evaluate_results(ctx.results, gt, ctx.scaffold)
            gi.print_evaluation_summary(ctx.eval_data, ctx.scaffold)
        else:
            gi.warn_line("Ground truth file could not be parsed — skipping evaluation.")
    else:
        gi.info_line("No reference list in the exam folder — skipped.")

    if ctx.through_step == 10:
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)


def _grade_step11_report(ctx: _GradeCtx, gi: SimpleNamespace) -> None:
    assert ctx.folder is not None and ctx.scaffold is not None and ctx.results is not None
    assert ctx.run_dir is not None
    gi.pipeline_step(11, "Report")
    if not ctx.no_report:
        output_tex = ctx.run_dir / "grade_report.tex"
        output_pdf = ctx.run_dir / "grade_report.pdf"
        title = f"{ctx.folder.name} — Grading Report"
        gi.generate_report(
            scaffold=ctx.scaffold,
            results=ctx.results,
            output_tex=output_tex,
            output_pdf=output_pdf,
            eval_data=ctx.eval_data,
            title=title,
        )
        gi.ok_line("Report saved.")
    else:
        gi.info_line("PDF report skipped (you turned it off).")
    if ctx.through_step == 11:
        gi.ok_line("Pipeline complete.")
        ctx.partial_stop_readme_step = ctx.through_step
        raise SystemExit(0)

    gi.ok_line("Grading pipeline finished.")
    ctx.pipeline_completed_ok = True


def _run(args: argparse.Namespace, timestamp: str) -> None:
    gi = _load_grade_imports()
    ctx = _GradeCtx(args=args, timestamp=timestamp)
    t0 = time.perf_counter()
    try:
        _grade_create_client(ctx, gi)
        _grade_step01_parse(ctx, gi)
        _grade_step02_folder(ctx, gi)
        _grade_step03_students(ctx, gi)
        _grade_step04_scaffold(ctx, gi)
        _grade_step05_clean_scan(ctx, gi)
        _grade_step06_page_assignment(ctx, gi)
        _grade_step07_detect(ctx, gi)
        _grade_step08_09_grade(ctx, gi)
        _grade_step10_eval(ctx, gi)
        _grade_step11_report(ctx, gi)
    finally:
        _print_grade_run_footer(ctx, gi, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
