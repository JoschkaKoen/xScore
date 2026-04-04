"""Generate a LaTeX/PDF grading report for the generic pipeline.

Sections:
  1. Exam Scaffold — questions, types, marks, correct answers
  2. Results Table — one row per student, one column per question
  3. Class Statistics — mean, median, high, low
  4. Ground Truth Evaluation — (only when GT is provided)

Compilation uses xelatex (for Unicode + CJK font support).
Auxiliary files (.aux, .log, .out) are cleaned up afterwards.
"""

from __future__ import annotations

import datetime
import statistics
import subprocess
from pathlib import Path

from shared.models import ExamScaffold, StudentResult


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _green(text: str) -> str:
    return r"\textcolor{OliveGreen}{" + _esc(str(text)) + "}"


def _red(text: str) -> str:
    return r"\textcolor{BrickRed}{" + _esc(str(text)) + "}"


def _amber(text: str) -> str:
    return r"\textcolor{Goldenrod}{" + _esc(str(text)) + "}"


def _pct_color_tex(pct: float, text: str) -> str:
    if pct >= 80:
        return _green(text)
    if pct >= 50:
        return _amber(text)
    return _red(text)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _scaffold_table(scaffold: ExamScaffold) -> str:
    rows = []
    for q in scaffold.gradable_questions:
        ans = _esc(q.correct_answer or "–")
        rows.append(
            f"    Q{_esc(q.number)} & {_esc(q.question_type.replace('_', ' '))} "
            f"& {q.marks} & {ans} \\\\"
        )
    body = "\n".join(rows)
    return rf"""
\section{{Exam Scaffold}}
\begin{{longtable}}{{l l r l}}
    \toprule
    \textbf{{Q\#}} & \textbf{{Type}} & \textbf{{Marks}} & \textbf{{Answer}} \\
    \midrule
    \endhead
{body}
    \midrule
    \multicolumn{{2}}{{l}}{{\textbf{{Total}}}} & \textbf{{{scaffold.total_marks}}} & \\
    \bottomrule
\end{{longtable}}
"""


def _results_table(
    results: list[StudentResult],
    scaffold: ExamScaffold,
    eval_data: dict | None,
) -> str:
    """Build the results longtable, colouring cells if GT evaluation is available."""
    q_nums = [q.number for q in scaffold.gradable_questions]
    max_m = results[0].max_marks if results else 0

    # Map student name → per-question eval dict
    eval_map: dict[str, dict[str, dict]] = {}
    if eval_data:
        for row in eval_data.get("per_student", []):
            eval_map[row["name"]] = row["per_question"]

    # Column spec: l (name) + c * questions + r (total) + r (pct)
    col_spec = "l " + "c " * len(q_nums) + "r r"
    q_headers = " & ".join(rf"\textbf{{Q{_esc(n)}}}" for n in q_nums)

    rows = []
    for r in results:
        name_cell = _esc(r.student_name)
        pq = eval_map.get(r.student_name, {})
        cells = []
        for q_num in q_nums:
            mark = r.marks_per_question.get(q_num)
            val_str = f"{mark:.1f}" if mark is not None else "–"
            if pq:
                info = pq.get(q_num)
                if info:
                    val_str = _green(val_str) if info["ok"] else _red(val_str)
                else:
                    val_str = _esc(val_str)
            else:
                val_str = _esc(val_str)
            cells.append(val_str)

        total_str = f"{r.total_marks:.1f}"
        pct = (r.total_marks / max_m * 100) if max_m else 0.0
        pct_str = f"{pct:.0f}\\%"
        total_cell = _pct_color_tex(pct, total_str)
        pct_cell = _pct_color_tex(pct, pct_str)

        rows.append(
            f"    {name_cell} & {' & '.join(cells)} & {total_cell} & {pct_cell} \\\\"
        )

    body = "\n".join(rows)
    return rf"""
\section{{Results}}
\begin{{longtable}}{{{col_spec}}}
    \toprule
    \textbf{{Student}} & {q_headers} & \textbf{{Total}} & \textbf{{\%}} \\
    \midrule
    \endhead
{body}
    \bottomrule
\end{{longtable}}
"""


def _stats_section(results: list[StudentResult]) -> str:
    if not results:
        return ""
    totals = [r.total_marks for r in results]
    max_m = results[0].max_marks or 1

    def row(label: str, val: float) -> str:
        pct = val / max_m * 100
        return rf"    {_esc(label)} & {val:.1f} & {pct:.1f}\% \\"

    lines = [
        row("Mean", statistics.mean(totals)),
        row("Median", statistics.median(totals)),
        row("Highest", max(totals)),
        row("Lowest", min(totals)),
    ]
    body = "\n".join(lines)
    return rf"""
\section{{Class Statistics}}
\begin{{tabular}}{{l r r}}
    \toprule
    \textbf{{Metric}} & \textbf{{Marks}} & \textbf{{\%}} \\
    \midrule
{body}
    \bottomrule
\end{{tabular}}
"""


def _evaluation_section(eval_data: dict, scaffold: ExamScaffold) -> str:
    q_nums = [q.number for q in scaffold.gradable_questions]
    col_spec = "l " + "c " * len(q_nums) + "r r"
    q_headers = " & ".join(rf"\textbf{{Q{_esc(n)}}}" for n in q_nums)

    rows = []
    for row in eval_data.get("per_student", []):
        name = _esc(row["name"])
        pq = row["per_question"]
        cells = []
        for q_num in q_nums:
            info = pq.get(q_num)
            if info is None:
                cells.append("–")
            elif info["ok"]:
                cells.append(_green(info["extracted"]))
            else:
                cells.append(
                    _red(info["extracted"]) + "/" + _esc(info["expected"])
                )
        acc = f"{row['accuracy_pct']:.0f}\\%"
        correct = f"{row['correct']}/{row['total']}"
        rows.append(f"    {name} & {' & '.join(cells)} & {acc} & {correct} \\\\")

    body = "\n".join(rows)
    overall = eval_data["overall_accuracy_pct"]
    overall_str = f"{eval_data['overall_correct']}/{eval_data['overall_total']} ({overall:.1f}\\%)"

    return rf"""
\section{{Ground Truth Evaluation}}

Overall accuracy: \textbf{{{overall_str}}}

\medskip
\begin{{longtable}}{{{col_spec}}}
    \toprule
    \textbf{{Student}} & {q_headers} & \textbf{{Acc}} & \textbf{{Score}} \\
    \midrule
    \endhead
{body}
    \bottomrule
\end{{longtable}}

\textit{{Cell format: extracted answer — wrong cells show extracted/expected.}}
"""


# ---------------------------------------------------------------------------
# Full document
# ---------------------------------------------------------------------------

def _full_document(
    title: str,
    scaffold: ExamScaffold,
    results: list[StudentResult],
    eval_data: dict | None,
) -> str:
    date_str = datetime.date.today().isoformat()

    scaffold_sec = _scaffold_table(scaffold)
    results_sec = _results_table(results, scaffold, eval_data)
    stats_sec = _stats_section(results)
    eval_sec = _evaluation_section(eval_data, scaffold) if eval_data else ""

    return rf"""\documentclass[a4paper,11pt]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage[dvipsnames]{{xcolor}}
\setCJKmainfont{{PingFang SC Regular}}[BoldFont=PingFang SC Semibold]

\title{{{_esc(title)}}}
\author{{Auto-Grader}}
\date{{{date_str}}}

\begin{{document}}
\maketitle
{scaffold_sec}
{results_sec}
{stats_sec}
{eval_sec}
\end{{document}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    scaffold: ExamScaffold,
    results: list[StudentResult],
    output_tex: Path,
    output_pdf: Path,
    eval_data: dict | None = None,
    title: str = "Exam Grading Report",
) -> bool:
    """Write LaTeX source to *output_tex* and compile to *output_pdf*.

    Returns ``True`` on success, ``False`` if xelatex fails (the .tex is
    still written for manual inspection).
    """
    from shared.terminal_ui import err_line, ok_line, tool_line

    output_tex.parent.mkdir(parents=True, exist_ok=True)

    doc = _full_document(title, scaffold, results, eval_data)
    output_tex.write_text(doc, encoding="utf-8")
    tool_line("report", "LaTeX source written.")

    # xelatex must be run in the output dir so aux files land there
    result = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", output_tex.name],
        cwd=str(output_tex.parent),
        capture_output=True,
        text=True,
    )

    # Clean up auxiliary files
    for ext in (".aux", ".log", ".out"):
        aux = output_tex.with_suffix(ext)
        if aux.exists():
            aux.unlink()

    if result.returncode != 0:
        err_line(f"xelatex failed (exit {result.returncode}).")
        tool_line("report", result.stdout[-800:] if result.stdout else "(no output)")
        return False

    ok_line("PDF report ready.")
    return True
