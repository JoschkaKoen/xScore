"""Build an ExamScaffold by parsing vector exam and answer-key PDFs (PyMuPDF).

No AI for structure: question regions follow left-margin numbering (Cambridge-style).
The list of questions is in **reading order** on the page(s); printed numbers may be out of order.
Results are cached as ``{folder}/scaffolds/scaffold_cache.json`` and reused if no PDF
is newer than the cache. Exam PDF figures are written under ``scaffold_images/exam``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.models import (
    BBox,
    ExamImage,
    ExamScaffold,
    McAnswerOption,
    Question,
    WritingArea,
    flatten_questions,
    gradable_questions,
)
from pipeline.pdf_parser import (
    merge_answers_into_scaffold,
    parse_answer_key_pdf,
    parse_exam_pdf,
    prepare_scaffold_image_dirs,
)
from pipeline.pdf_parser.content import normalize_multiple_choice_tree
from pipeline.scaffold_overlay import write_scaffold_boxes_pdf


SCHEMA_VERSION = 14


def _find_exam_pdf(folder: Path) -> Path:
    pdfs = [
        f
        for f in folder.glob("*.pdf")
        if ("raw" in f.name.lower() or "exam" in f.name.lower())
        and "answer" not in f.name.lower()
        and "scan" not in f.name.lower()
    ]
    if not pdfs:
        raise FileNotFoundError(f"No raw exam PDF found in {folder}")
    return pdfs[0]


def _find_answer_pdf(folder: Path) -> Path | None:
    answer_pdfs = [f for f in folder.glob("*.pdf") if "answer" in f.name.lower()]
    return answer_pdfs[0] if answer_pdfs else None


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------

# Keep cache JSON readable: 1 fractional digit is sufficient for PDF-point coordinates.
_JSON_COORD_DECIMALS = 1


def _round_coord(v: float) -> float:
    return round(float(v), _JSON_COORD_DECIMALS)


def _bbox_to_dict(b: BBox) -> dict:
    return {
        "x0": _round_coord(b.x0),
        "y0": _round_coord(b.y0),
        "x1": _round_coord(b.x1),
        "y1": _round_coord(b.y1),
        "page": b.page,
    }


def _bbox_from_dict(d: dict) -> BBox:
    return BBox(
        float(d["x0"]),
        float(d["y0"]),
        float(d["x1"]),
        float(d["y1"]),
        int(d["page"]),
    )


def _img_to_dict(im: ExamImage) -> dict:
    return {"bbox": _bbox_to_dict(im.bbox), "path": im.path}


def _img_from_dict(d: dict) -> ExamImage:
    return ExamImage(bbox=_bbox_from_dict(d["bbox"]), path=d["path"])


def _wa_to_dict(w: WritingArea) -> dict:
    return {"bbox": _bbox_to_dict(w.bbox), "kind": w.kind}


def _wa_from_dict(d: dict) -> WritingArea:
    return WritingArea(bbox=_bbox_from_dict(d["bbox"]), kind=d["kind"])


def question_to_dict(q: Question) -> dict[str, Any]:
    """Serialize for cache JSON; omit nulls and empty collections (sparse)."""
    opts_dicts = [{"letter": o.letter, "text": o.text} for o in q.answer_options]
    d: dict[str, Any] = {
        "number": q.number,
        "question_type": q.question_type,
        "text": q.text,
        "marks": q.marks,
        "bbox": _bbox_to_dict(q.bbox),
    }
    if opts_dicts:
        d["answer_options"] = opts_dicts
    if q.equation_blank_bboxes:
        d["equation_blank_bboxes"] = [_bbox_to_dict(b) for b in q.equation_blank_bboxes]
    if q.images:
        d["images"] = [_img_to_dict(i) for i in q.images]
    if q.writing_areas:
        d["writing_areas"] = [_wa_to_dict(w) for w in q.writing_areas]
    if q.subquestions:
        d["subquestions"] = [question_to_dict(s) for s in q.subquestions]
    if q.correct_answer is not None and str(q.correct_answer).strip():
        d["correct_answer"] = q.correct_answer
    if q.marking_criteria is not None and str(q.marking_criteria).strip():
        d["marking_criteria"] = q.marking_criteria
    if q.answer_images:
        d["answer_images"] = [_img_to_dict(i) for i in q.answer_images]
    return d


def question_from_dict(d: dict) -> Question:
    # Migrate v1 cache (AI scaffold)
    text = d.get("text")
    if text is None:
        text = d.get("content_summary", "")
    bbox_d = d.get("bbox")
    if not bbox_d:
        bbox_d = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0, "page": 1}
    ao = [
        McAnswerOption(letter=str(x["letter"]), text=str(x.get("text") or ""))
        for x in (d.get("answer_options") or [])
        if isinstance(x, dict) and x.get("letter")
    ]
    ca = d.get("correct_answer")
    if ca is None or (isinstance(ca, str) and not str(ca).strip()):
        # Migrate older caches that stored answer_key_text instead of correct_answer
        leg = d.get("answer_key_text")
        if leg and str(leg).strip():
            ca = str(leg).strip()
    return Question(
        number=str(d["number"]),
        question_type=d.get("question_type", "short_answer"),
        text=text,
        marks=int(d.get("marks", 1)),
        bbox=_bbox_from_dict(bbox_d),
        equation_blank_bboxes=[_bbox_from_dict(x) for x in d.get("equation_blank_bboxes") or []],
        images=[_img_from_dict(x) for x in d.get("images") or []],
        writing_areas=[_wa_from_dict(x) for x in d.get("writing_areas") or []],
        subquestions=[question_from_dict(s) for s in d.get("subquestions") or []],
        correct_answer=ca,
        marking_criteria=d.get("marking_criteria"),
        answer_images=[_img_from_dict(x) for x in d.get("answer_images") or []],
        answer_options=ao,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _scaffolds_dir(folder: Path) -> Path:
    return folder / "scaffolds"


def _cache_path(folder: Path) -> Path:
    return _scaffolds_dir(folder) / "scaffold_cache.json"


def _legacy_cache_path(folder: Path) -> Path:
    """Pre-layout: cache lived at the exam folder root."""
    return folder / "scaffold_cache.json"


def _effective_cache_path(folder: Path) -> Path | None:
    new = _cache_path(folder)
    if new.exists():
        return new
    leg = _legacy_cache_path(folder)
    if leg.exists():
        return leg
    return None


def _is_cache_valid(folder: Path) -> bool:
    cache = _effective_cache_path(folder)
    if cache is None:
        return False
    cache_mtime = cache.stat().st_mtime
    for pdf in folder.glob("*.pdf"):
        if pdf.stat().st_mtime > cache_mtime:
            return False
    return True


def _load_cache(folder: Path) -> ExamScaffold:
    path = _effective_cache_path(folder)
    if path is None:
        raise FileNotFoundError(f"No scaffold cache in {folder}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = [question_from_dict(q) for q in data["questions"]]
    flat = flatten_questions(questions)
    total = int(data.get("total_marks", 0))
    if not total and questions:
        total = sum(q.marks for q in gradable_questions(questions))
    scaffold = ExamScaffold(
        questions=questions,
        total_marks=total,
        page_count=int(data.get("page_count", 0)),
        raw_description=data.get("raw_description", ""),
    )
    # One-time move: root `scaffold_cache.json` → `scaffolds/scaffold_cache.json`
    if path.resolve() == _legacy_cache_path(folder).resolve():
        _save_cache(folder, scaffold)
    return scaffold


def _save_cache(folder: Path, scaffold: ExamScaffold) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "questions": [question_to_dict(q) for q in scaffold.questions],
        "total_marks": scaffold.total_marks,
        "page_count": scaffold.page_count,
        "raw_description": scaffold.raw_description,
    }
    out = _cache_path(folder)
    _scaffolds_dir(folder).mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    leg = _legacy_cache_path(folder)
    if leg.exists() and leg.resolve() != out.resolve():
        try:
            leg.unlink()
        except OSError:
            pass


def build_scaffold(folder: Path, client: Any | None = None, dpi: int = 200) -> ExamScaffold:
    """Build (or load from cache) the ExamScaffold for the exam in *folder*.

    *client* is optional and unused (kept for backward compatibility with callers).
    *dpi* is unused; parsing is vector-based.
    """
    _ = client, dpi
    if _is_cache_valid(folder):
        try:
            print("[scaffold] Loading scaffold from cache …")
            return _load_cache(folder)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            print("[scaffold] Cache incompatible or corrupt — rebuilding …")

    exam_pdf = _find_exam_pdf(folder)
    prepare_scaffold_image_dirs(folder)

    print(f"[scaffold] Parsing exam PDF (vector): {exam_pdf.name} …")
    questions = parse_exam_pdf(exam_pdf, folder)
    if not questions:
        raise RuntimeError(
            "No questions detected in exam PDF. Check that the file is a vector paper "
            "with Cambridge-style left-margin question numbers."
        )

    ans = _find_answer_pdf(folder)
    if ans is not None:
        print(f"[scaffold] Parsing answer key PDF (vector): {ans.name} …")
        amap, table_answers, printed_mc = parse_answer_key_pdf(ans, folder)
        merge_answers_into_scaffold(
            questions,
            amap,
            table_model_answers=table_answers,
            printed_mc_letters=printed_mc,
        )
    else:
        print("[scaffold] No answer key PDF found — correct_answer left empty.")

    for q in questions:
        normalize_multiple_choice_tree(q)

    import fitz

    doc = fitz.open(exam_pdf)
    try:
        page_count = len(doc)
    finally:
        doc.close()

    flat = flatten_questions(questions)
    leaves = gradable_questions(questions)
    total_marks = sum(q.marks for q in leaves)
    raw_description = (
        f"{len(questions)} top-level, {len(leaves)} gradable parts, {total_marks} marks; "
        + ", ".join(f"Q{q.number}({q.marks}m)" for q in leaves[:24])
        + (" …" if len(leaves) > 24 else "")
    )

    scaffold = ExamScaffold(
        questions=questions,
        total_marks=total_marks,
        page_count=page_count,
        raw_description=raw_description,
    )
    _save_cache(folder, scaffold)
    out_pdf, n_rects, n_pages = write_scaffold_boxes_pdf(exam_pdf, questions)
    print(
        f"[scaffold] Scaffold built: {len(questions)} top-level questions, "
        f"{len(leaves)} gradable parts, {total_marks} total marks."
    )
    print(
        f"[scaffold] Bounding-box overlay: {out_pdf.name} "
        f"({n_rects} rectangles on {n_pages} page(s))."
    )
    return scaffold
