"""Public entry points: parse exam / answer PDFs, merge, prepare output dirs."""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz

from pipeline.shared.models import Question, flatten_questions
from pipeline.scaffold.pdf_parser.build import build_questions_from_segments
from pipeline.scaffold.pdf_parser.config import DEFAULT_PARSER_CONFIG, ParserConfig
from pipeline.scaffold.pdf_parser.content import (
    ensure_multiple_choice_options_parsed,
    infer_answer_fields,
    normalize_exam_scientific_text,
)
from pipeline.scaffold.pdf_parser.layout import cell_margin_band, cell_scales
from pipeline.scaffold.pdf_parser.regions import (
    clip_for_segment,
    clip_for_text_segment,
    find_question_positions,
    iter_region_segments,
)


def _scaffold_key_from_table_label_line(line: str) -> str | None:
    """Map ``11(a)``, ``11(c)(i)`` … to scaffold numbers ``11a``, ``11ci`` …."""
    s = line.strip()
    m = re.match(r"^(\d+)\(c\)\(iii\)\s*$", s, re.I)
    if m:
        return f"{m.group(1)}ciii"
    m = re.match(r"^(\d+)\(c\)\(ii\)\s*$", s, re.I)
    if m:
        return f"{m.group(1)}cii"
    m = re.match(r"^(\d+)\(c\)\(i\)\s*$", s, re.I)
    if m:
        return f"{m.group(1)}ci"
    m = re.match(r"^(\d+)\(([a-z])\)\s*$", s, re.I)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    return None


# ``Question 38 (Answer: A)`` lines — full-document scan (order matches scaffold MC leaves).
_PRINTED_MC_ANSWER_RE = re.compile(
    r"Question\s+(\d+(?:_\d+)?)\s*\(\s*Answer\s*:\s*([A-Da-d])\s*\)",
    re.I,
)


def printed_mc_answer_letters_from_doc(doc: fitz.Document) -> list[str]:
    """MC letters in PDF text order from printed mark-scheme lines."""
    parts: list[str] = []
    for page in doc:
        t = page.get_text()
        if t:
            parts.append(t)
    blob = "\n".join(parts)
    return [m.group(2).upper() for m in _PRINTED_MC_ANSWER_RE.finditer(blob)]


def _is_mark_scheme_section_break_line(stripped: str) -> bool:
    """Stop capturing a table answer before the next exercise block or MC section."""
    if stripped.startswith("IGCSE "):
        return True
    if re.match(r"^Answers\s+Q", stripped, re.I):
        return True
    if re.match(r"^Question\s+\d+\s*\(\s*Answer\s*:", stripped, re.I):
        return True
    return False


def parse_mark_scheme_table_answers(doc: fitz.Document) -> dict[str, str]:
    """Parse two-column mark-scheme tables: left cell ``N(a)`` / ``N(c)(i)``, right = model answer."""
    parts: list[str] = []
    for page in doc:
        t = page.get_text()
        if t:
            parts.append(t)
    raw_lines = "\n".join(parts).splitlines()
    out: dict[str, str] = {}
    i = 0
    while i < len(raw_lines):
        stripped = raw_lines[i].strip()
        key = _scaffold_key_from_table_label_line(stripped)
        if key is None:
            i += 1
            continue
        i += 1
        acc: list[str] = []
        while i < len(raw_lines):
            stripped = raw_lines[i].strip()
            if _scaffold_key_from_table_label_line(stripped) is not None:
                break
            if _is_mark_scheme_section_break_line(stripped):
                break
            acc.append(stripped)
            i += 1
        body = normalize_exam_scientific_text("\n".join(acc).strip())
        if body:
            out[key] = body
    return out


def parse_exam_pdf(
    pdf_path: Path,
    exam_folder: Path,
    cfg: ParserConfig = DEFAULT_PARSER_CONFIG,
    *,
    artifact_dir: Path | None = None,
) -> list[Question]:
    """Parse blank exam vector PDF; write images under artifact_dir/scaffold_images/."""
    from pipeline.shared.exam_paths import exam_artifact_dir

    root = artifact_dir or exam_artifact_dir(exam_folder)
    doc = fitz.open(pdf_path)
    try:
        positions = find_question_positions(doc, cfg)
        if not positions:
            return []
        segments = iter_region_segments(doc, positions, cfg)
        return build_questions_from_segments(doc, segments, root, cfg)
    finally:
        doc.close()


def parse_answer_key_pdf(
    pdf_path: Path,
    exam_folder: Path,
    cfg: ParserConfig = DEFAULT_PARSER_CONFIG,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    """Parse answer key PDF.

    Returns ``(margin-question map, table_model_answers, printed_mc_letters)``.
    *table_model_answers* maps scaffold ids (e.g. ``11a``) to model text from ``11(a)`` rows.
    *printed_mc_letters* lists ``Question N (Answer: X)`` letters in document order.
    """
    doc = fitz.open(pdf_path)
    result: dict[str, dict[str, Any]] = {}
    try:
        positions = find_question_positions(doc, cfg)
        if not positions:
            return {}, parse_mark_scheme_table_answers(doc), printed_mc_answer_letters_from_doc(doc)
        segments = iter_region_segments(doc, positions, cfg)
        by_q: dict[str, list[tuple[int, float, float, fitz.Rect, int, float, bool, bool]]] = defaultdict(
            list
        )
        order: list[str] = []
        for qid, pidx, y0, y1, cell, printed_raw, num_x1, snap_top, snap_bottom in segments:
            if qid not in order:
                order.append(qid)
            by_q[qid].append((pidx, y0, y1, cell, printed_raw, num_x1, snap_top, snap_bottom))

        for qid in order:
            text_parts: list[str] = []
            for pidx, y0, y1, cell, _printed_raw, num_x1, snap_top, snap_bottom in by_q[qid]:
                page = doc[pidx]
                y0c = float(cell.y0) if snap_top else y0
                y1c = float(cell.y1) if snap_bottom else y1
                clip = clip_for_segment(doc, pidx, y0c, y1c, cfg, cell)
                mt, _mb = cell_margin_band(cell, cfg)
                _sx, sy = cell_scales(cell)
                pad_above = min(cfg.padding_above * sy, cfg.text_clip_pad_above_pt * sy)
                text_y0 = max(y0 - pad_above, mt)
                text_clip = clip_for_text_segment(doc, pidx, text_y0, y1, cfg, cell, num_x1)
                chunk = page.get_text("text", clip=text_clip).strip()
                if chunk:
                    text_parts.append(chunk)

            full_text = normalize_exam_scientific_text("\n\n".join(text_parts).strip())
            ca, mc = infer_answer_fields(full_text)
            result[qid] = {
                "full_text": full_text,
                "correct_answer": ca,
                "marking_criteria": mc,
                "answer_images": [],
            }
        table = parse_mark_scheme_table_answers(doc)
        printed_mc = printed_mc_answer_letters_from_doc(doc)
        return result, table, printed_mc
    finally:
        doc.close()


def merge_answers_into_scaffold(
    questions: list[Question],
    answer_map: dict[str, dict[str, Any]],
    table_model_answers: dict[str, str] | None = None,
    printed_mc_letters: list[str] | None = None,
) -> None:
    def walk(qs: list[Question]) -> None:
        for q in qs:
            entry = answer_map.get(q.number)
            if entry:
                q.correct_answer = entry.get("correct_answer")
                q.marking_criteria = entry.get("marking_criteria")
                q.answer_images = list(entry.get("answer_images") or [])
                ca = q.correct_answer or ""
                if len(ca) == 1 and ca.upper() in "ABCD":
                    q.question_type = "multiple_choice"
            walk(q.subquestions)

    walk(questions)
    if table_model_answers:
        for q in flatten_questions(questions):
            body = table_model_answers.get(q.number)
            if not body:
                continue
            q.correct_answer = body.strip()
            q.marking_criteria = None

        def clear_superseded(node: Question) -> None:
            for sq in node.subquestions:
                clear_superseded(sq)
            if not node.subquestions or node.number in table_model_answers:
                return
            desc = flatten_questions(node.subquestions)
            if any(d.number in table_model_answers for d in desc):
                node.correct_answer = None
                node.marking_criteria = None

        for q in questions:
            clear_superseded(q)
    for q in questions:
        ensure_multiple_choice_options_parsed(q)

    if printed_mc_letters:
        mc_leaves = [
            x
            for x in flatten_questions(questions)
            if x.question_type == "multiple_choice" and not x.subquestions
        ]
        for q, letter in zip(mc_leaves, printed_mc_letters):
            q.correct_answer = letter


def prepare_scaffold_image_dirs(artifact_dir: Path) -> Path:
    """Create empty ``scaffold_images`` under *artifact_dir*; remove prior tree there.

    Answer-key PDFs are not rasterized here (they are text-only mark schemes).
    """
    base = artifact_dir / "scaffold_images"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    return base
