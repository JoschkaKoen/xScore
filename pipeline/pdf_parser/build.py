"""Assemble ``Question`` objects from region segments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import fitz

from pipeline.models import BBox, ExamImage, Question, WritingArea
from pipeline.pdf_parser.config import ParserConfig
from pipeline.pdf_parser.content import (
    detect_writing_areas,
    extract_images,
    infer_marks,
    infer_question_type,
    safe_image_stem,
    strip_question_tree_stems,
)
from pipeline.pdf_parser.regions import clip_for_segment, clip_for_text_segment
from pipeline.pdf_parser.subparts import maybe_split_written_subquestions


def build_questions_from_segments(
    doc: fitz.Document,
    segments: list[tuple[str, int, float, float, fitz.Rect, int, float]],
    exam_folder: Path,
    cfg: ParserConfig,
) -> list[Question]:
    by_q: dict[str, list[tuple[int, float, float, fitz.Rect, int, float]]] = defaultdict(list)
    order: list[str] = []
    for qid, pidx, y0, y1, cell, printed_raw, num_x1 in segments:
        if qid not in order:
            order.append(qid)
        by_q[qid].append((pidx, y0, y1, cell, printed_raw, num_x1))

    img_counter = [0]
    questions: list[Question] = []

    for qid in order:
        segs = by_q[qid]
        text_parts: list[str] = []
        all_images: list[ExamImage] = []
        all_areas: list[WritingArea] = []
        first_bbox: BBox | None = None
        stem_base = safe_image_stem(qid)

        for pidx, y0, y1, cell, printed_raw, num_x1 in segs:
            page = doc[pidx]
            clip = clip_for_segment(doc, pidx, y0, y1, cfg, cell)
            text_clip = clip_for_text_segment(doc, pidx, y0, y1, cfg, cell, num_x1)
            page_1 = pidx + 1
            if first_bbox is None:
                first_bbox = BBox(clip.x0, clip.y0, clip.x1, clip.y1, page_1)

            chunk = page.get_text("text", clip=text_clip).strip()
            if chunk:
                text_parts.append(chunk)

            all_areas.extend(detect_writing_areas(page, clip, page_1, cfg))

            stem = f"q{stem_base}_p{page_1}"
            imgs = extract_images(page, clip, exam_folder, "exam", stem, page_1, img_counter)
            all_images.extend(imgs)

        full_text = "\n\n".join(text_parts).strip()
        marks = infer_marks(full_text)
        qtype = infer_question_type(full_text)

        q = Question(
            number=qid,
            question_type=qtype,
            text=full_text,
            marks=marks,
            bbox=first_bbox or BBox(0, 0, 0, 0, 1),
            images=all_images,
            writing_areas=all_areas,
            subquestions=[],
        )
        q = maybe_split_written_subquestions(q, doc, segs, cfg)
        strip_question_tree_stems(q)
        questions.append(q)

    return questions
