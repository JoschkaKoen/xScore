"""Stem heuristics, drawings, and embedded images."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from pipeline.models import BBox, ExamImage, Question, WritingArea
from pipeline.pdf_parser.config import ParserConfig


def detect_writing_areas(
    page: fitz.Page, clip: fitz.Rect, page_1based: int, cfg: ParserConfig
) -> list[WritingArea]:
    areas: list[WritingArea] = []
    for d in page.get_drawings():
        r = fitz.Rect(d.get("rect", (0, 0, 0, 0)))
        if r.is_empty or not r.intersects(clip):
            continue
        ir = r & clip
        if ir.is_empty:
            continue
        if ir.width >= cfg.box_min_width and ir.height >= cfg.box_min_height:
            areas.append(
                WritingArea(
                    bbox=BBox(ir.x0, ir.y0, ir.x1, ir.y1, page_1based),
                    kind="box",
                )
            )
        elif ir.height <= cfg.line_max_height and ir.width >= cfg.line_min_width:
            areas.append(
                WritingArea(
                    bbox=BBox(ir.x0, ir.y0, ir.x1, ir.y1, page_1based),
                    kind="lines",
                )
            )
    return areas


def extract_images(
    page: fitz.Page,
    clip: fitz.Rect,
    exam_folder: Path,
    subdir: str,
    stem: str,
    page_1based: int,
    img_counter: list[int],
) -> list[ExamImage]:
    out_dir = exam_folder / "scaffold_images" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    found: list[ExamImage] = []
    for item in page.get_images(full=True):
        xref = item[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            if not r.intersects(clip):
                continue
            ir = r & clip
            img_counter[0] += 1
            name = f"{stem}_{img_counter[0]}.png"
            abs_path = out_dir / name
            try:
                pix = page.get_pixmap(clip=ir, dpi=150)
                pix.save(str(abs_path))
            except Exception:
                continue
            rel = f"scaffold_images/{subdir}/{name}"
            found.append(
                ExamImage(
                    bbox=BBox(ir.x0, ir.y0, ir.x1, ir.y1, page_1based),
                    path=rel,
                )
            )
    return found


def marks_from_square_brackets(text: str) -> int | None:
    """Sum all ``[N]`` (1–40) in *text*; strips ``[Total: …]`` regions first. Returns ``None`` if none."""
    if not text or not text.strip():
        return None
    stripped = re.sub(r"(?is)\[Total:\s*\d+\s*\]", "", text)
    total = 0
    for m in re.finditer(r"\[\s*(\d+)\s*\]", stripped):
        v = int(m.group(1))
        if 1 <= v <= 40:
            total += v
    if total == 0:
        return None
    return min(total, 99)


def infer_marks(text: str) -> int:
    """Marks from Cambridge-style ``[N]`` brackets; then ``[N marks]`` / line-ending ``(N)``."""
    sq = marks_from_square_brackets(text)
    if sq is not None:
        return max(1, sq)
    m = re.search(r"\[(\d+)\s*mark", text, re.I)
    if m:
        v = int(m.group(1))
        return max(1, v) if v <= 40 else 1
    m2 = re.search(r"\((\d+)\)\s*$", text, re.M)
    if m2:
        v2 = int(m2.group(1))
        return max(1, v2) if 1 <= v2 <= 20 else 1
    return 1


def strip_exam_mark_indicators(text: str) -> str:
    """Remove mark cues (``[2]``, ``[Total: 5]``, ``[3 marks]``) from stem text after marks are inferred."""
    if not text or not text.strip():
        return text
    t = text
    t = re.sub(r"(?is)\[Total:\s*\d+\s*\]", "", t)
    t = re.sub(r"(?is)\[\s*\d+\s*marks?\s*\]", "", t)

    def _blank_bracket_num(m: re.Match[str]) -> str:
        try:
            v = int(m.group(1))
        except ValueError:
            return m.group(0)
        if 1 <= v <= 40:
            return ""
        return m.group(0)

    t = re.sub(r"\[\s*(\d+)\s*\]", _blank_bracket_num, t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_question_tree_stems(q: Question) -> None:
    """Apply :func:`strip_exam_mark_indicators` to *q* and every nested subquestion."""
    q.text = strip_exam_mark_indicators(q.text)
    for sq in q.subquestions:
        strip_question_tree_stems(sq)


def rollup_question_marks(q: Question) -> int:
    """Set each non-leaf ``marks`` to the sum of its descendants; leaves unchanged."""
    if not q.subquestions:
        return q.marks
    s = sum(rollup_question_marks(sq) for sq in q.subquestions)
    q.marks = s
    return s


def infer_question_type(text: str) -> str:
    tl = text.lower()
    if "multiple choice" in tl or re.search(r"(?m)^\s*[A-Da-d]\s{1,4}\d", text):
        return "multiple_choice"
    if "calculate" in tl or "show your working" in tl:
        return "calculation"
    if len(text) > 400:
        return "long_answer"
    return "short_answer"


def infer_answer_fields(full_text: str) -> tuple[str | None, str | None]:
    stripped = full_text.strip()
    if not stripped:
        return None, None
    lines = stripped.splitlines()
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()

    m = re.match(r"^[A-Da-d]$", first)
    if m:
        return m.group(0).upper(), (rest or None)

    m2 = re.match(r"^(\d{1,2})\s+([A-Da-d])\s*$", first)
    if m2:
        return m2.group(2).upper(), (rest or None)

    if len(first) <= 4:
        m3 = re.search(r"\b([A-Da-d])\b", first)
        if m3 and re.match(r"^[\d\sA-Da-d.]+$", first):
            return m3.group(1).upper(), (rest or None)

    return (first[:4000] if first else None), (rest or None)


def safe_image_stem(qid: str) -> str:
    return re.sub(r"[^\w\-]", "_", qid)
