"""Public entry points: parse exam / answer PDFs, merge, prepare output dirs."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz

from pipeline.models import Question
from pipeline.pdf_parser.build import build_questions_from_segments
from pipeline.pdf_parser.config import DEFAULT_PARSER_CONFIG, ParserConfig
from pipeline.pdf_parser.content import extract_images, infer_answer_fields, safe_image_stem
from pipeline.pdf_parser.regions import (
    clip_for_segment,
    clip_for_text_segment,
    find_question_positions,
    iter_region_segments,
)


def parse_exam_pdf(
    pdf_path: Path, exam_folder: Path, cfg: ParserConfig = DEFAULT_PARSER_CONFIG
) -> list[Question]:
    """Parse blank exam vector PDF; write images under exam_folder/scaffold_images/exam/."""
    doc = fitz.open(pdf_path)
    try:
        positions = find_question_positions(doc, cfg)
        if not positions:
            return []
        segments = iter_region_segments(doc, positions, cfg)
        return build_questions_from_segments(doc, segments, exam_folder, cfg)
    finally:
        doc.close()


def parse_answer_key_pdf(
    pdf_path: Path,
    exam_folder: Path,
    cfg: ParserConfig = DEFAULT_PARSER_CONFIG,
) -> dict[str, dict[str, Any]]:
    """Parse answer key PDF."""
    doc = fitz.open(pdf_path)
    result: dict[str, dict[str, Any]] = {}
    try:
        positions = find_question_positions(doc, cfg)
        if not positions:
            return {}
        segments = iter_region_segments(doc, positions, cfg)
        img_counter = [0]
        by_q: dict[str, list[tuple[int, float, float, fitz.Rect, int, float]]] = defaultdict(list)
        order: list[str] = []
        for qid, pidx, y0, y1, cell, printed_raw, num_x1 in segments:
            if qid not in order:
                order.append(qid)
            by_q[qid].append((pidx, y0, y1, cell, printed_raw, num_x1))

        for qid in order:
            text_parts: list[str] = []
            answer_images = []
            stem_base = safe_image_stem(qid)
            for pidx, y0, y1, cell, _printed_raw, num_x1 in by_q[qid]:
                page = doc[pidx]
                clip = clip_for_segment(doc, pidx, y0, y1, cfg, cell)
                text_clip = clip_for_text_segment(doc, pidx, y0, y1, cfg, cell, num_x1)
                page_1 = pidx + 1
                chunk = page.get_text("text", clip=text_clip).strip()
                if chunk:
                    text_parts.append(chunk)
                stem = f"ans{stem_base}_p{page_1}"
                answer_images.extend(
                    extract_images(page, clip, exam_folder, "answers", stem, page_1, img_counter)
                )

            full_text = "\n\n".join(text_parts).strip()
            ca, mc = infer_answer_fields(full_text)
            result[qid] = {
                "full_text": full_text,
                "correct_answer": ca,
                "marking_criteria": mc,
                "answer_images": answer_images,
            }
        return result
    finally:
        doc.close()


def merge_answers_into_scaffold(questions: list[Question], answer_map: dict[str, dict[str, Any]]) -> None:
    def walk(qs: list[Question]) -> None:
        for q in qs:
            entry = answer_map.get(q.number)
            if entry:
                q.answer_key_text = entry.get("full_text") or None
                q.correct_answer = entry.get("correct_answer")
                q.marking_criteria = entry.get("marking_criteria")
                q.answer_images = list(entry.get("answer_images") or [])
                ca = q.correct_answer or ""
                if len(ca) == 1 and ca.upper() in "ABCD":
                    q.question_type = "multiple_choice"
            walk(q.subquestions)

    walk(questions)


def prepare_scaffold_image_dirs(exam_folder: Path) -> tuple[Path, Path]:
    """Create ``scaffold_images/exam`` and ``scaffold_images/answers``; clear old assets."""
    base = exam_folder / "scaffold_images"
    exam_d = base / "exam"
    ans_d = base / "answers"
    if base.exists():
        shutil.rmtree(base)
    exam_d.mkdir(parents=True)
    ans_d.mkdir(parents=True)
    return exam_d, ans_d
