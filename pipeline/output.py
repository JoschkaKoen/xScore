"""Well-formatted terminal output for the grading pipeline."""

from __future__ import annotations

import statistics
from typing import Sequence

from extraction.reporting import Colors

from pipeline.models import ExamScaffold, PageAssignment, StudentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(width: int = 60) -> str:
    return "─" * width


def _pct_color(pct: float) -> str:
    if pct >= 80:
        return Colors.GREEN
    if pct >= 50:
        return Colors.YELLOW
    return Colors.RED


# ---------------------------------------------------------------------------
# Scaffold summary
# ---------------------------------------------------------------------------

def print_scaffold_summary(scaffold: ExamScaffold) -> None:
    print(f"\n{'═' * 60}")
    print(f"  EXAM SCAFFOLD  —  {len(scaffold.questions)} questions, {scaffold.total_marks} total marks")
    print(f"{'═' * 60}")
    col_w = (55 - 10 - 20 - 8) // 2
    print(f"  {'#':<6} {'Type':<20} {'Marks':>5}  {'Answer':<15}")
    print(f"  {_bar(56)}")
    for q in scaffold.questions:
        ans = q.correct_answer or "–"
        print(f"  Q{q.number:<5} {q.question_type:<20} {q.marks:>5}  {ans:<15}")
    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Page assignment summary
# ---------------------------------------------------------------------------

def print_page_summary(page_map: list[PageAssignment], students: list[str]) -> None:
    print(f"\n{'═' * 60}")
    print(f"  PAGE ASSIGNMENT  —  {len(page_map)} student(s) identified")
    print(f"{'═' * 60}")

    found_names = {a.student_name for a in page_map}

    for assignment in page_map:
        conf_color = Colors.GREEN if assignment.confidence == "high" else Colors.YELLOW
        pages_str = ", ".join(str(p) for p in assignment.page_numbers)
        print(
            f"  {assignment.student_name:<20} "
            f"pages [{pages_str}]  "
            f"{conf_color}({assignment.confidence}){Colors.RESET}"
        )

    not_found = [s for s in students if s not in found_names]
    if not_found:
        print(f"\n  {Colors.RED}Students in roster NOT found in scan:{Colors.RESET}")
        for name in not_found:
            print(f"    – {name}")

    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Exercise detection summary
# ---------------------------------------------------------------------------

def print_exercise_summary(exercise_map: dict[str, list[str]]) -> None:
    print(f"\n{'═' * 60}")
    print(f"  ANSWERED EXERCISES")
    print(f"{'═' * 60}")
    for name, questions in exercise_map.items():
        q_str = ", ".join(questions) if questions else "none"
        print(f"  {name:<20}  Q: {q_str}")
    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(results: list[StudentResult], scaffold: ExamScaffold) -> None:
    if not results:
        print("  No results to display.")
        return

    q_nums = [q.number for q in scaffold.questions]
    max_name = max((len(r.student_name) for r in results), default=10)
    col_w = 5  # per-question column width

    header_q = "".join(f"  Q{n:<{col_w}}" for n in q_nums)
    print(f"\n{'═' * 60}")
    print(f"  RESULTS TABLE")
    print(f"{'═' * 60}")
    print(f"  {'Student':<{max_name}}  {header_q}  {'Total':>7}  {'%':>6}")
    print(f"  {_bar(max_name + len(header_q) + 18)}")

    for r in results:
        q_marks = "".join(
            f"  {str(r.marks_per_question.get(n, '–')):>{col_w + 2}}"
            for n in q_nums
        )
        pct = (r.total_marks / r.max_marks * 100) if r.max_marks else 0.0
        color = _pct_color(pct)
        print(
            f"  {r.student_name:<{max_name}}  {q_marks}  "
            f"{color}{r.total_marks:>6.1f}  {pct:>5.1f}%{Colors.RESET}"
        )

    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Grand summary
# ---------------------------------------------------------------------------

def print_grand_summary(results: list[StudentResult]) -> None:
    if not results:
        return

    totals = [r.total_marks for r in results]
    max_m = results[0].max_marks if results else 0

    mean = statistics.mean(totals)
    median = statistics.median(totals)
    hi = max(totals)
    lo = min(totals)

    def fmt(val: float) -> str:
        pct = (val / max_m * 100) if max_m else 0
        color = _pct_color(pct)
        return f"{color}{val:.1f} ({pct:.0f}%){Colors.RESET}"

    print(f"{'═' * 60}")
    print(f"  CLASS STATISTICS  —  {len(results)} student(s) graded")
    print(f"{'═' * 60}")
    print(f"  Mean:    {fmt(mean)}")
    print(f"  Median:  {fmt(median)}")
    print(f"  Highest: {fmt(hi)}")
    print(f"  Lowest:  {fmt(lo)}")
    print(f"{'═' * 60}\n")
