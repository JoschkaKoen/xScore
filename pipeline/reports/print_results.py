"""Well-formatted terminal output for the grading pipeline."""

from __future__ import annotations

import re
import shutil
import statistics
import textwrap

from extraction.reporting import Colors

from pipeline.shared.models import ExamScaffold, PageAssignment, StudentResult
from pipeline.shared.terminal_ui import icon, paint, rule, BOLD, CYAN


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

_BULLET_ONLY_LINE = frozenset(
    {
        "•",
        "·",
        "▪",
        "‣",
        "-",
        "*",
        "○",
    }
)

# Unicode bullets only — avoid "-" / "*" here (false positives in prose).
_BULLET_CLASS = "•·▪‣"
_NEWLINE_AFTER_BULLET = re.compile(
    rf"([{_BULLET_CLASS}])\s*\n(?!\s*(?:[{_BULLET_CLASS}]))",
)


def _collapse_newline_after_bullet(text: str) -> str:
    """Turn ``•\\nitem`` into ``• item``; keep ``\\n`` before ``•`` (new list item)."""
    prev = None
    while prev != text:
        prev = text
        text = _NEWLINE_AFTER_BULLET.sub(r"\1 ", text)
    return text


def _normalize_scaffold_answer_lines(text: str) -> str:
    """Join mark-scheme lines where the bullet sits alone on a line before the item text."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()
        if stripped in _BULLET_ONLY_LINE:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if nxt and nxt not in _BULLET_ONLY_LINE:
                    out.append(f"• {nxt}")
                    i = j + 1
                    continue
        out.append(raw_line)
        i += 1
    return "\n".join(out)


def _rejoin_lonely_bullet_wrap_lines(lines: list[str]) -> list[str]:
    """After textwrap, merge a line that is only a bullet with the following non-empty line."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "":
            out.append(line)
            i += 1
            continue
        st = line.strip()
        if st in _BULLET_ONLY_LINE:
            j = i + 1
            while j < len(lines) and lines[j] == "":
                j += 1
            if j < len(lines):
                nxt_st = lines[j].strip()
                if nxt_st and nxt_st not in _BULLET_ONLY_LINE:
                    out.append(line.rstrip() + " " + lines[j].lstrip())
                    i = j + 1
                    continue
        out.append(line)
        i += 1
    return out


def print_scaffold_summary(scaffold: ExamScaffold) -> None:
    """Print gradable questions with wrapped model answers (multi-line safe)."""
    print()
    print(rule("═"))
    leaves = scaffold.gradable_questions
    print(
        paint(
            f"  {icon('gear')}  EXAM SCAFFOLD  —  {len(scaffold.questions)} top-level, "
            f"{len(leaves)} gradable parts, {scaffold.total_marks} total marks",
            CYAN,
            BOLD,
        )
    )
    print(rule("═"))
    # Fixed-width meta columns; answer text wraps in remaining terminal width.
    w_q, w_ty, w_m = 10, 24, 5
    meta_len = 2 + w_q + w_ty + w_m + 2
    term_w = max(64, shutil.get_terminal_size((96, 20)).columns)
    ans_width = max(32, term_w - meta_len)

    print(
        f"  {'Question':<{w_q}}{'Type':<{w_ty}}{'Marks':>{w_m}}  "
        "Model answer (wrapped to terminal width)"
    )
    print(f"  {_bar(min(term_w - 4, 92))}")

    def _type_cell(qt: str) -> str:
        label = qt.replace("_", " ")
        return label if len(label) <= w_ty else label[: w_ty - 1] + "…"

    for q in leaves:
        raw = (q.correct_answer or "").strip() or "–"
        raw = re.sub(r"\r\n?", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = _collapse_newline_after_bullet(raw)
        raw = _normalize_scaffold_answer_lines(raw)

        wrapped_lines: list[str] = []
        for para in raw.split("\n"):
            p = para.strip()
            if not p:
                if wrapped_lines and wrapped_lines[-1] != "":
                    wrapped_lines.append("")
                continue
            chunk = textwrap.wrap(
                p,
                width=ans_width,
                break_long_words=True,
                break_on_hyphens=True,
            )
            wrapped_lines.extend(chunk if chunk else [""])

        wrapped_lines = _rejoin_lonely_bullet_wrap_lines(wrapped_lines)

        if not wrapped_lines:
            wrapped_lines = ["–"]

        q_label = f"Q{q.number}"
        first_meta = f"  {q_label:<{w_q}}{_type_cell(q.question_type):<{w_ty}}{q.marks:>{w_m}}  "
        indent = " " * len(first_meta)

        for i, line in enumerate(wrapped_lines):
            if i == 0:
                print(first_meta + line)
            else:
                print(indent + line)

    print(rule("═"))
    print()


# ---------------------------------------------------------------------------
# Page assignment summary
# ---------------------------------------------------------------------------

def print_page_summary(page_map: list[PageAssignment], students: list[str]) -> None:
    print()
    print(rule("═"))
    print(
        paint(
            f"  {icon('users')}  PAGE ASSIGNMENT  —  {len(page_map)} student(s) identified",
            CYAN,
            BOLD,
        )
    )
    print(rule("═"))

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

    print(rule("═"))
    print()


# ---------------------------------------------------------------------------
# Exercise detection summary
# ---------------------------------------------------------------------------

def print_exercise_summary(exercise_map: dict[str, list[str]]) -> None:
    print()
    print(rule("═"))
    print(paint(f"  {icon('search')}  ANSWERED EXERCISES", CYAN, BOLD))
    print(rule("═"))
    for name, questions in exercise_map.items():
        q_str = ", ".join(questions) if questions else "none"
        print(f"  {name:<20}  Q: {q_str}")
    print(rule("═"))
    print()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(results: list[StudentResult], scaffold: ExamScaffold) -> None:
    if not results:
        print("  No results to display.")
        return

    q_nums = [q.number for q in scaffold.gradable_questions]
    max_name = max((len(r.student_name) for r in results), default=10)
    col_w = 5  # per-question column width

    header_q = "".join(f"  Q{n:<{col_w}}" for n in q_nums)
    print()
    print(rule("═"))
    print(paint(f"  {icon('chart')}  RESULTS TABLE", CYAN, BOLD))
    print(rule("═"))
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

    print(rule("═"))
    print()


# ---------------------------------------------------------------------------
# Grand summary
# ---------------------------------------------------------------------------

def print_evaluation_summary(eval_data: dict, scaffold: ExamScaffold) -> None:
    """Print per-student and overall accuracy against ground truth."""
    overall_pct = eval_data["overall_accuracy_pct"]
    overall_str = (
        f"{eval_data['overall_correct']}/{eval_data['overall_total']}"
        f"  ({overall_pct:.1f}%)"
    )
    color = _pct_color(overall_pct)

    print()
    print(rule("═"))
    print(paint(f"  {icon('chart')}  GROUND TRUTH EVALUATION", CYAN, BOLD))
    print(rule("═"))
    print(f"  Overall accuracy: {color}{overall_str}{Colors.RESET}")
    print()

    q_nums = [q.number for q in scaffold.gradable_questions]
    max_name = max((len(r["name"]) for r in eval_data["per_student"]), default=10)

    # Header
    q_hdr = "  ".join(f"Q{n}" for n in q_nums)
    print(f"  {'Student':<{max_name}}  {q_hdr:<{len(q_hdr)}}  {'Acc':>6}")
    print(f"  {_bar(max_name + len(q_hdr) + 12)}")

    for row in eval_data["per_student"]:
        pq = row["per_question"]
        cells = []
        for q_num in q_nums:
            info = pq.get(q_num)
            if info is None:
                cells.append(" –")
            elif info["ok"]:
                cells.append(f"{Colors.GREEN}{info['extracted']:>2}{Colors.RESET}")
            else:
                cells.append(f"{Colors.RED}{info['extracted']:>2}{Colors.RESET}")
        cell_str = "  ".join(cells)
        acc_pct = row["accuracy_pct"]
        acc_color = _pct_color(acc_pct)
        print(
            f"  {row['name']:<{max_name}}  {cell_str}  "
            f"{acc_color}{row['correct']}/{row['total']} ({acc_pct:.0f}%){Colors.RESET}"
        )

    print(rule("═"))
    print()


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

    print(rule("═"))
    print(
        paint(
            f"  {icon('chart')}  CLASS STATISTICS  —  {len(results)} student(s) graded",
            CYAN,
            BOLD,
        )
    )
    print(rule("═"))
    print(f"  Mean:    {fmt(mean)}")
    print(f"  Median:  {fmt(median)}")
    print(f"  Highest: {fmt(hi)}")
    print(f"  Lowest:  {fmt(lo)}")
    print(rule("═"))
    print()
