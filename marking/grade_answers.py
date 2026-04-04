"""Grade student answers against the ExamScaffold (Step G).

Three modes driven by TaskInstruction.task_type:

  check_mc      — one Kimi call per MC question per student; binary (0 or 1 mark).
  check_answers — check_mc for MC questions + marking-criteria call for written ones.
  count_marks   — teacher-marked in red; one Kimi call per page, tally red scores.

Granularity: one AI call per question per student for check_* modes.
This gives focused, accurate results; batching can be added later.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .kimi_helpers import kimi_image_call, page_to_jpeg_b64, parse_json_safe
from shared.models import ExamScaffold, PageAssignment, Question, StudentResult, TaskInstruction


# ---------------------------------------------------------------------------
# Mode: check_mc
# ---------------------------------------------------------------------------

def _prompt_mc(question: Question) -> str:
    answer_hint = f" The correct answer is {question.correct_answer}." if question.correct_answer else ""
    return (
        f"Look at Question {question.number} on this exam page. "
        f"What letter (A/B/C/D) did the student mark for this question? "
        f"Return ONLY: {{\"answer\": \"B\"}} or {{\"answer\": \"?\"}} if unreadable."
        f"{answer_hint}"
    )


def _grade_mc(
    client: Any,
    pages: list,
    question: Question,
) -> tuple[str, float]:
    """Return (student_answer, marks_awarded)."""
    correct = (question.correct_answer or "").upper().strip()

    for page in pages:
        img_b64 = page_to_jpeg_b64(page)
        raw = kimi_image_call(client, img_b64, _prompt_mc(question))
        data = parse_json_safe(raw)
        answer = str(data.get("answer", "?")).upper().strip()
        if answer not in ("", "?"):
            marks = float(question.marks) if answer == correct else 0.0
            return answer, marks
        time.sleep(0.15)

    return "?", 0.0


# ---------------------------------------------------------------------------
# Mode: check_answers (written questions)
# ---------------------------------------------------------------------------

def _prompt_written(question: Question) -> str:
    criteria = question.marking_criteria or question.correct_answer or "(see mark scheme)"
    return (
        f"Look at Question {question.number} on this exam page "
        f"(max {question.marks} mark{'s' if question.marks != 1 else ''}). "
        f"Marking criteria: {criteria}. "
        f"What mark should this student receive? "
        f"Return ONLY: {{\"answer\": \"student's answer summary\", \"marks\": 1.5}}"
    )


def _grade_written(
    client: Any,
    pages: list,
    question: Question,
) -> tuple[str, float]:
    for page in pages:
        img_b64 = page_to_jpeg_b64(page)
        raw = kimi_image_call(client, img_b64, _prompt_written(question), max_tokens=256)
        data = parse_json_safe(raw)
        answer = str(data.get("answer", "")).strip() or "?"
        try:
            marks = min(float(data.get("marks", 0)), float(question.marks))
        except (TypeError, ValueError):
            marks = 0.0
        if answer not in ("", "?"):
            return answer, marks
        time.sleep(0.15)

    return "?", 0.0


# ---------------------------------------------------------------------------
# Mode: count_marks
# ---------------------------------------------------------------------------

_COUNT_PROMPT = """\
Look at this exam page for marks written in red by the teacher. \
For each question you can see a red mark/score, extract it.

Return ONLY:
{"marks": {"1": 2, "2a": 1, "38": 0}}

If no red marks are visible, return:
{"marks": {}}
"""


def _count_marks_on_page(client: Any, page) -> dict[str, float]:
    img_b64 = page_to_jpeg_b64(page)
    raw = kimi_image_call(client, img_b64, _COUNT_PROMPT, max_tokens=256)
    data = parse_json_safe(raw)
    raw_marks = data.get("marks", {})
    result: dict[str, float] = {}
    for k, v in raw_marks.items():
        try:
            result[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _filter_students(
    page_map: list[PageAssignment],
    instruction: TaskInstruction,
) -> list[PageAssignment]:
    sf = instruction.student_filter
    if sf.mode == "specific" and sf.names:
        lower_names = {n.lower() for n in sf.names}
        return [a for a in page_map if a.student_name.lower() in lower_names]
    if sf.mode == "first_n" and sf.n > 0:
        return page_map[: sf.n]
    return page_map


def grade_students(
    cleaned_pdf: Path,
    page_map: list[PageAssignment],
    exercise_map: dict[str, list[str]],
    scaffold: ExamScaffold,
    instruction: TaskInstruction,
    client: Any | None = None,
    *,
    pages: list | None = None,
) -> list[StudentResult]:
    """Grade all students and return a ``StudentResult`` for each.

    If *client* is None it is created via ``KimiProvider.create_client()``.
    *pages*: optional pre-rendered page images at ``instruction.dpi`` (skips ``convert_from_path``).
    """
    from pdf2image import convert_from_path

    if client is None:
        from extraction.providers.kimi import KimiProvider
        client = KimiProvider.create_client()
    if client is None:
        raise RuntimeError("No Kimi client available for grading.")

    dpi = instruction.dpi
    task = instruction.task_type
    assignments = _filter_students(page_map, instruction)

    from shared.terminal_ui import info_line, tool_line

    if pages is None:
        tool_line("grade", f"Rendering pages @ {dpi} DPI …")
        all_pages = convert_from_path(str(cleaned_pdf), dpi=dpi, thread_count=os.cpu_count() or 4)
    else:
        all_pages = pages

    leaves = scaffold.gradable_questions
    mc_questions = [q for q in leaves if q.question_type == "multiple_choice"]
    all_questions = leaves

    results: list[StudentResult] = []

    for assignment in assignments:
        name = assignment.student_name
        student_pages = [all_pages[n - 1] for n in assignment.page_numbers if 1 <= n <= len(all_pages)]
        if not student_pages:
            continue

        attempted_nums = set(exercise_map.get(name, [q.number for q in all_questions]))
        answers: dict[str, str] = {}
        marks_per_q: dict[str, float] = {}

        info_line(f"Grading {name} ({len(student_pages)} page(s), mode={task}) …")

        if task == "count_marks":
            for page in student_pages:
                page_marks = _count_marks_on_page(client, page)
                for q_num, m in page_marks.items():
                    marks_per_q[q_num] = marks_per_q.get(q_num, 0.0) + m
                time.sleep(0.15)

        elif task in ("check_mc", "check_answers"):
            questions_to_grade = mc_questions if task == "check_mc" else all_questions
            for q in questions_to_grade:
                if q.number not in attempted_nums:
                    answers[q.number] = "-"
                    marks_per_q[q.number] = 0.0
                    continue

                if q.question_type == "multiple_choice":
                    ans, marks = _grade_mc(client, student_pages, q)
                else:
                    ans, marks = _grade_written(client, student_pages, q)

                answers[q.number] = ans
                marks_per_q[q.number] = marks
                info_line(f"Q{q.number}: {ans}  →  {marks}/{q.marks}")
                time.sleep(0.15)

        total = sum(marks_per_q.values())
        max_marks = sum(q.marks for q in scaffold.gradable_questions)

        results.append(StudentResult(
            student_name=name,
            page_numbers=assignment.page_numbers,
            answers=answers,
            marks_per_question=marks_per_q,
            total_marks=total,
            max_marks=max_marks,
        ))

    return results
