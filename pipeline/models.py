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
    # Optional explicit exam folder path (from prompt); lower priority than CLI --folder
    folder_path: str | None = None
    skip_clean_scan: bool = False
    force_clean_scan: bool = False
    rescaffold: bool = False
    through_step: int | None = None  # 1–11: exit after README pipeline step N
    no_report: bool = False


@dataclass
class BBox:
    """Bounding box in PDF points; *page* is 1-based (first page = 1)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page: int


@dataclass
class ExamImage:
    bbox: BBox
    path: str


@dataclass
class WritingArea:
    bbox: BBox
    kind: str  # "box" | "lines"


@dataclass
class McAnswerOption:
    """One row in a multiple-choice stem (Cambridge-style letter on its own line)."""

    letter: str  # "A" … "D"
    text: str


@dataclass
class Question:
    number: str                 # hierarchical label: "9", "9a", "9ai", "9aii"; duplicate mains "38_2"
    question_type: str          # "multiple_choice" | "short_answer" | "calculation" | "long_answer"
    text: str                   # stem only for MC (options in answer_options); full text otherwise
    marks: int
    bbox: BBox                  # primary region (first segment of multi-page questions)
    images: list[ExamImage] = field(default_factory=list)
    equation_blank_bboxes: list[BBox] = field(default_factory=list)  # one per "label = …… [n]" line
    writing_areas: list[WritingArea] = field(default_factory=list)
    subquestions: list[Question] = field(default_factory=list)
    correct_answer: str | None = None
    marking_criteria: str | None = None
    answer_images: list[ExamImage] = field(default_factory=list)
    answer_options: list[McAnswerOption] = field(default_factory=list)  # MC only

    @property
    def content_summary(self) -> str:
        """Backward compatibility: first line or truncated text."""
        line = self.text.strip().split("\n", 1)[0].strip()
        return line[:200] + ("…" if len(line) > 200 else "")


def flatten_questions(questions: list[Question]) -> list[Question]:
    """Depth-first list of this node and all nested subquestions."""
    out: list[Question] = []
    for q in questions:
        out.append(q)
        out.extend(flatten_questions(q.subquestions))
    return out


def gradable_questions(questions: list[Question]) -> list[Question]:
    """Leaf questions only (parts that carry marks); skips parent nodes that have subquestions."""
    out: list[Question] = []
    for q in questions:
        if q.subquestions:
            out.extend(gradable_questions(q.subquestions))
        else:
            out.append(q)
    return out


@dataclass
class ExamScaffold:
    questions: list[Question]
    total_marks: int
    page_count: int = 0
    raw_description: str = ""

    @property
    def all_questions(self) -> list[Question]:
        """Every scaffold node in document reading order (parents and nested subparts)."""
        return flatten_questions(self.questions)

    @property
    def gradable_questions(self) -> list[Question]:
        """Leaf parts only — use for summing exam marks and per-part grading."""
        return gradable_questions(self.questions)


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
