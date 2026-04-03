"""Assemble ``Question`` objects from region segments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import fitz

from pipeline.shared.models import BBox, ExamImage, McAnswerOption, Question
from pipeline.scaffold.pdf_parser.config import ParserConfig
from pipeline.scaffold.pdf_parser.content import (
    adjust_leaf_bboxes_after_previous_exercise,
    extract_images,
    infer_marks,
    infer_question_type,
    normalize_multiple_choice_tree,
    safe_image_stem,
    split_mc_options_from_stem,
    strip_question_tree_stems,
)
from pipeline.scaffold.pdf_parser.answer_fields import assign_answer_field_bboxes
from pipeline.scaffold.pdf_parser.layout import (
    apply_subpage_vertical_snaps,
    cell_margin_band,
    cell_scales,
    expand_bbox_to_subpage_width,
)
from pipeline.scaffold.pdf_parser.regions import clip_for_segment, clip_for_text_segment
from pipeline.scaffold.pdf_parser.subparts import maybe_split_written_subquestions


def build_questions_from_segments(
    doc: fitz.Document,
    segments: list[tuple[str, int, float, float, fitz.Rect, int, float, bool, bool]],
    artifact_dir: Path,
    cfg: ParserConfig,
) -> list[Question]:
    by_q: dict[str, list[tuple[int, float, float, fitz.Rect, int, float, bool, bool]]] = defaultdict(
        list
    )
    order: list[str] = []
    for qid, pidx, y0, y1, cell, printed_raw, num_x1, snap_top, snap_bottom in segments:
        if qid not in order:
            order.append(qid)
        by_q[qid].append((pidx, y0, y1, cell, printed_raw, num_x1, snap_top, snap_bottom))

    img_counter = [0]
    questions: list[Question] = []

    for qid in order:
        segs = by_q[qid]
        text_parts: list[str] = []
        all_images: list[ExamImage] = []
        first_bbox: BBox | None = None
        stem_base = safe_image_stem(qid)

        for pidx, y0, y1, cell, printed_raw, num_x1, snap_top, snap_bottom in segs:
            page = doc[pidx]
            y0c = float(cell.y0) if snap_top else y0
            y1c = float(cell.y1) if snap_bottom else y1
            clip = clip_for_segment(doc, pidx, y0c, y1c, cfg, cell)
            mt, _mb = cell_margin_band(cell, cfg)
            _sx, sy = cell_scales(cell)
            pad_above = min(cfg.padding_above * sy, cfg.text_clip_pad_above_pt * sy)
            text_y0 = max(y0 - pad_above, mt)
            text_clip = clip_for_text_segment(doc, pidx, text_y0, y1, cfg, cell, num_x1)
            page_1 = pidx + 1
            if first_bbox is None:
                first_bbox = BBox(clip.x0, clip.y0, clip.x1, clip.y1, page_1)

            chunk = page.get_text("text", clip=text_clip).strip()
            if chunk:
                text_parts.append(chunk)

            stem = f"q{stem_base}_p{page_1}"
            for im in extract_images(page, clip, artifact_dir, stem, page_1, img_counter, cfg):
                all_images.append(
                    ExamImage(
                        bbox=expand_bbox_to_subpage_width(doc, im.bbox),
                        path=im.path,
                    )
                )

        full_text = "\n\n".join(text_parts).strip()
        marks = infer_marks(full_text)
        qtype = infer_question_type(full_text)
        stem_text = full_text
        answer_opts: list[McAnswerOption] = []
        if qtype == "multiple_choice":
            stem_text, answer_opts = split_mc_options_from_stem(full_text)
            if not answer_opts:
                stem_text = full_text

        q = Question(
            number=qid,
            question_type=qtype,
            text=stem_text,
            marks=marks,
            answer_options=answer_opts,
            bbox=first_bbox or BBox(0, 0, 0, 0, 1),
            images=all_images,
            subquestions=[],
        )
        q = maybe_split_written_subquestions(q, doc, segs, cfg)
        strip_question_tree_stems(q)
        # Snaps must run before answer-field inference so the final leaf bboxes
        # (including the last leaf in each cell, stretched to the cell bottom) are
        # in place when equation-blank detection scans for "label = …… [n]" lines.
        apply_subpage_vertical_snaps(
            doc, cfg, q, segs[0][3], segs[0][6], segs[-1][7]
        )
        normalize_multiple_choice_tree(q)
        assign_answer_field_bboxes(doc, cfg, q)
        questions.append(q)

    # Pull each non-first leaf bbox.y0 to sit just below the previous exercise's
    # last text line, then re-infer equation blanks on the updated bboxes.
    adjust_leaf_bboxes_after_previous_exercise(doc, cfg, questions)

    return questions
