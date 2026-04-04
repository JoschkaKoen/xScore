"""Well-formatted terminal output for the grading pipeline (Rich)."""

from __future__ import annotations

import re
import shutil
import statistics
import textwrap

from rich import box
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from shared.models import ExamScaffold, PageAssignment, StudentResult
from shared.terminal_ui import get_console, icon


def _pct_style(pct: float) -> str:
    if pct >= 80:
        return "green"
    if pct >= 50:
        return "yellow"
    return "red"


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
    c = get_console()
    leaves = scaffold.gradable_questions
    w_q, w_ty, w_m = 10, 24, 5
    meta_len = 2 + w_q + w_ty + w_m + 2
    term_w = max(64, shutil.get_terminal_size((96, 20)).columns)
    ans_width = max(32, term_w - meta_len)

    def _type_cell(qt: str) -> str:
        label = qt.replace("_", " ")
        return label if len(label) <= w_ty else label[: w_ty - 1] + "…"

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=(
            f"  {icon('gear')}  Exam scaffold  —  {len(leaves)} questions, {scaffold.total_marks} marks"
        ),
        title_style="bold cyan",
        expand=False,
    )
    table.add_column("Question", width=w_q, overflow="ellipsis")
    table.add_column("Type", width=w_ty, overflow="ellipsis")
    table.add_column("Marks", justify="right", width=w_m)
    table.add_column("Model answer (wrapped)", overflow="fold", max_width=ans_width + 8)

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

        answer_cell = Text("\n".join(wrapped_lines))
        table.add_row(
            f"Q{q.number}",
            _type_cell(q.question_type),
            str(q.marks),
            answer_cell,
        )

    c.print()
    c.print(table)
    c.print()


# ---------------------------------------------------------------------------
# Page assignment summary
# ---------------------------------------------------------------------------


def print_page_summary(page_map: list[PageAssignment], students: list[str]) -> None:
    c = get_console()
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"{icon('users')}  PAGE ASSIGNMENT  —  {len(page_map)} student(s) identified",
        title_style="bold cyan",
    )
    table.add_column("Student", min_width=20)
    table.add_column("Pages", overflow="fold")
    table.add_column("Conf.", justify="center", width=8)

    found_names = {a.student_name for a in page_map}

    for assignment in page_map:
        pages_str = ", ".join(str(p) for p in assignment.page_numbers)
        conf_style = "green" if assignment.confidence == "high" else "yellow"
        table.add_row(
            assignment.student_name,
            pages_str,
            Text(f"({assignment.confidence})", style=conf_style),
        )

    c.print()
    c.print(Panel(table, border_style="dim cyan"))

    not_found = [s for s in students if s not in found_names]
    if not_found:
        c.print()
        c.print(f"[red]  Students in roster NOT found in scan:[/]")
        for name in not_found:
            c.print(f"    – {name}")
    c.print()


# ---------------------------------------------------------------------------
# Exercise detection summary
# ---------------------------------------------------------------------------


def print_exercise_summary(exercise_map: dict[str, list[str]]) -> None:
    c = get_console()
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"{icon('search')}  ANSWERED EXERCISES",
        title_style="bold cyan",
    )
    table.add_column("Student", min_width=20)
    table.add_column("Questions", overflow="fold")

    for name, questions in exercise_map.items():
        q_str = ", ".join(questions) if questions else "none"
        table.add_row(name, q_str)

    c.print()
    c.print(Panel(table, border_style="dim cyan"))
    c.print()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_results_table(results: list[StudentResult], scaffold: ExamScaffold) -> None:
    c = get_console()
    if not results:
        c.print("[dim]  No results to display.[/]")
        return

    q_nums = [q.number for q in scaffold.gradable_questions]
    col_w = 5

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"{icon('chart')}  RESULTS TABLE",
        title_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Student", min_width=12, no_wrap=True)
    for n in q_nums:
        table.add_column(f"Q{n}", justify="right", width=col_w + 1)
    table.add_column("Total", justify="right", width=8)
    table.add_column("%", justify="right", width=7)

    for r in results:
        pct = (r.total_marks / r.max_marks * 100) if r.max_marks else 0.0
        st = _pct_style(pct)
        row_cells: list[str | Text] = [r.student_name]
        for n in q_nums:
            row_cells.append(str(r.marks_per_question.get(n, "–")))
        row_cells.append(Text(f"{r.total_marks:6.1f}", style=st))
        row_cells.append(Text(f"{pct:5.1f}%", style=st))
        table.add_row(*row_cells)

    c.print()
    c.print(Panel(table, border_style="dim cyan"))
    c.print()


# ---------------------------------------------------------------------------
# Grand summary
# ---------------------------------------------------------------------------


def print_evaluation_summary(eval_data: dict, scaffold: ExamScaffold) -> None:
    """Print per-student and overall accuracy against ground truth."""
    c = get_console()
    overall_pct = eval_data["overall_accuracy_pct"]
    overall_str = (
        f"{eval_data['overall_correct']}/{eval_data['overall_total']}"
        f"  ({overall_pct:.1f}%)"
    )
    st_overall = _pct_style(overall_pct)

    c.print()
    c.print(Rule(f"{icon('chart')}  GROUND TRUTH EVALUATION", style="bold cyan"))
    c.print(
        Text.assemble(
            "  Overall accuracy: ",
            (overall_str, st_overall),
        )
    )
    c.print()

    q_nums = [q.number for q in scaffold.gradable_questions]

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Student", min_width=12, no_wrap=True)
    for n in q_nums:
        table.add_column(f"Q{n}", justify="center", width=4)
    table.add_column("Acc", justify="right", min_width=14)

    for row in eval_data["per_student"]:
        pq = row["per_question"]
        cells: list[str | Text] = [row["name"]]
        for q_num in q_nums:
            info = pq.get(q_num)
            if info is None:
                cells.append("–")
            elif info["ok"]:
                cells.append(Text(f"{info['extracted']:>2}", style="green"))
            else:
                cells.append(Text(f"{info['extracted']:>2}", style="red"))
        acc_pct = row["accuracy_pct"]
        acc_st = _pct_style(acc_pct)
        cells.append(
            Text(
                f"{row['correct']}/{row['total']} ({acc_pct:.0f}%)",
                style=acc_st,
            )
        )
        table.add_row(*cells)

    c.print(Panel(table, border_style="dim cyan"))
    c.print()


def print_grand_summary(results: list[StudentResult]) -> None:
    if not results:
        return

    c = get_console()
    totals = [r.total_marks for r in results]
    max_m = results[0].max_marks if results else 0

    mean = statistics.mean(totals)
    median = statistics.median(totals)
    hi = max(totals)
    lo = min(totals)

    def fmt_cell(val: float) -> Text:
        pct = (val / max_m * 100) if max_m else 0
        st = _pct_style(pct)
        return Text(f"{val:.1f} ({pct:.0f}%)", style=st)

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"{icon('chart')}  CLASS STATISTICS  —  {len(results)} student(s) graded",
        title_style="bold cyan",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Mean", fmt_cell(mean))
    table.add_row("Median", fmt_cell(median))
    table.add_row("Highest", fmt_cell(hi))
    table.add_row("Lowest", fmt_cell(lo))

    c.print(Panel(table, border_style="dim cyan"))
    c.print()
