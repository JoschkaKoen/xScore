"""Ground truth discovery, loading, and evaluation for the generic pipeline.

Ground truth files live inside the exam folder. Supported filenames (checked
in order):
    ground_truth.txt  |  ground_truth.tsv  |  Ground Truth   |
    Ground Truth.txt  |  answers.txt

File format (tab- or space-separated):
    Line 1 : header — first token is "Name" (or anything); remaining tokens
              are question numbers that correspond to scaffold question numbers.
              If line 1 looks like student data (no recognisable header), the
              question numbers are inferred from the scaffold in order.
    Lines 2+: student rows — name followed by one value per question.

Values can be:
    - Single letters (A/B/C/D) → treated as MC answers, compared by equality.
    - Numbers (int or float)   → treated as marks, compared by equality.
    - Mixed columns are handled per-column.

Example (MC exam):
    Name   Q38_LT  Q39_L  Q40_L  Q38_LB  Q39_R  Q40_R
    Yuze   A       D      B      C       A      C

Example (marked exam):
    Name   1   2   3
    Alice  2   1   0
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.models import ExamScaffold, StudentResult


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_CANDIDATE_NAMES = [
    "ground_truth.txt",
    "ground_truth.tsv",
    "Ground Truth ",       # trailing space — matches existing convention
    "Ground Truth",
    "Ground Truth.txt",
    "answers.txt",
]


def find_ground_truth_file(folder: Path) -> Path | None:
    """Return the path to a ground truth file in *folder*, or None."""
    for name in _CANDIDATE_NAMES:
        p = folder / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_HEADER_TOKENS = {"name", "student", "students", "names", "#"}


def _is_data_row(tokens: list[str]) -> bool:
    """Return True if this row looks like student data (name + answer/mark values)."""
    if len(tokens) < 2:
        return False
    # Values (tokens[1:]) must be single letters (A-D) or numbers
    return all(
        (t.upper() in {"A", "B", "C", "D", "?"}) or _is_number(t)
        for t in tokens[1:]
    )


def load_ground_truth(
    folder: Path,
    scaffold: ExamScaffold | None = None,
) -> dict[str, dict[str, str]] | None:
    """Load the ground truth file from *folder*.

    Returns ``{student_name: {question_number: value}}`` or ``None`` if no
    file is found.

    Question numbers are taken from:
    1. A header row, if the file has one (first column = "Name" / similar,
       remaining columns = question numbers).
    2. The scaffold's question numbers in order (if scaffold is provided).
    3. Positional indices "1", "2", … as fallback.

    Label-only lines (e.g. the filename "Ground Truth " printed inside the
    file) are automatically skipped.
    """
    path = find_ground_truth_file(folder)
    if path is None:
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    # Strip comments and blank lines
    lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return None

    # Determine column separator (tab-separated takes priority)
    sep = "\t" if any("\t" in ln for ln in lines) else None

    def split_line(line: str) -> list[str]:
        return [t.strip() for t in (line.split(sep) if sep else line.split()) if t.strip()]

    # Separate potential header row from data rows
    # A header row has non-data tokens in columns 1+ (e.g. "Q38_LT", "Truth")
    # A title/label row has only 1-2 tokens that aren't answer values
    q_numbers: list[str] = []
    data_lines: list[str] = []

    for i, line in enumerate(lines):
        tokens = split_line(line)
        if not tokens:
            continue
        if _is_data_row(tokens):
            data_lines.append(line)
        elif i == 0 or not data_lines:
            # Could be a header row (question numbers in columns 1+)
            # Only treat as a real header if there are enough columns
            # to match the data rows that follow
            if len(tokens) >= 2 and tokens[0].lower() in _HEADER_TOKENS:
                q_numbers = [t for t in tokens[1:]]
            # Otherwise it's a label/title line — skip it
        # Other non-data rows (mid-file labels) are also skipped

    if not data_lines:
        return None

    # If we got no q_numbers from a header, infer from scaffold or use indices
    if not q_numbers:
        n_cols = max(len(split_line(ln)) - 1 for ln in data_lines)
        if scaffold is not None:
            q_numbers = [q.number for q in scaffold.gradable_questions[:n_cols]]
        else:
            q_numbers = [str(i + 1) for i in range(n_cols)]

    gt: dict[str, dict[str, str]] = {}
    for line in data_lines:
        parts = split_line(line)
        if len(parts) < 2:
            continue
        name = parts[0]
        values = parts[1:]
        row: dict[str, str] = {}
        for q_num, val in zip(q_numbers, values):
            row[q_num] = val
        gt[name] = row

    return gt if gt else None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_results(
    results: list[StudentResult],
    ground_truth: dict[str, dict[str, str]],
    scaffold: ExamScaffold,
) -> dict:
    """Compare grading results against ground truth.

    Returns a dict with:
        per_student: list of {name, correct, total, accuracy_pct,
                               per_question: {q_num: {extracted, expected, ok}}}
        overall_correct: int
        overall_total:   int
        overall_accuracy_pct: float
    """
    from extraction.ground_truth import fuzzy_match_name

    gt_names = list(ground_truth.keys())
    q_nums = [q.number for q in scaffold.gradable_questions]

    per_student = []
    overall_correct = 0
    overall_total = 0

    for result in results:
        matched_gt = fuzzy_match_name(result.student_name, gt_names)
        if matched_gt is None:
            continue

        gt_row = ground_truth[matched_gt]
        per_question: dict[str, dict] = {}
        correct = 0
        total = 0

        for q_num in q_nums:
            extracted = result.marks_per_question.get(q_num)
            extracted_ans = result.answers.get(q_num)
            expected = gt_row.get(q_num)

            if expected is None:
                continue

            total += 1

            # Decide comparison mode for this column
            if _is_number(expected):
                # Marks comparison
                try:
                    ok = abs(float(extracted or 0) - float(expected)) < 0.01
                    ext_str = str(extracted) if extracted is not None else "?"
                except (TypeError, ValueError):
                    ok = False
                    ext_str = "?"
            else:
                # Answer string comparison (MC letter)
                ext_str = (extracted_ans or "?").upper().strip()
                ok = ext_str == expected.upper().strip() and ext_str not in ("", "?")

            if ok:
                correct += 1

            per_question[q_num] = {
                "extracted": ext_str,
                "expected": expected,
                "ok": ok,
            }

        overall_correct += correct
        overall_total += total

        per_student.append({
            "name": result.student_name,
            "matched_gt": matched_gt,
            "correct": correct,
            "total": total,
            "accuracy_pct": (correct / total * 100) if total else 0.0,
            "per_question": per_question,
        })

    overall_pct = (overall_correct / overall_total * 100) if overall_total else 0.0
    return {
        "per_student": per_student,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy_pct": overall_pct,
    }
