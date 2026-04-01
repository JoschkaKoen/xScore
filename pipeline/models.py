"""Data structures for the generic grading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StudentFilter:
    mode: str = "all"           # "all" | "specific" | "first_n"
    names: list[str] = field(default_factory=list)
    n: int = 0


@dataclass
class TaskInstruction:
    task_type: str              # "count_marks" | "check_mc" | "check_answers"
    student_filter: StudentFilter = field(default_factory=StudentFilter)
    dpi: int = 400
    folder_hint: str | None = None


@dataclass
class Question:
    number: str                 # "1", "2a", "38"
    question_type: str          # "multiple_choice" | "short_answer" | "calculation" | "long_answer"
    content_summary: str
    marks: int
    correct_answer: str | None = None
    marking_criteria: str | None = None


@dataclass
class ExamScaffold:
    questions: list[Question]
    total_marks: int
    raw_description: str        # AI's full text description, kept for debugging


@dataclass
class PageAssignment:
    student_name: str
    page_numbers: list[int]
    confidence: str             # "high" | "medium" | "low"


@dataclass
class StudentResult:
    student_name: str
    page_numbers: list[int]
    answers: dict[str, str]                 # question_number → student's answer
    marks_per_question: dict[str, float]
    total_marks: float
    max_marks: float
