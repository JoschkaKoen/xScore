"""Build an ExamScaffold by parsing vector exam and answer-key PDFs (PyMuPDF).

No AI for structure: question regions follow left-margin numbering (Cambridge-style).
The list of questions is in **reading order** on the page(s); printed numbers may be out of order.
Results are cached under ``{artifact_dir}/scaffolds/scaffold_cache.json`` (default
``output/<exam_stem>/`` via :func:`pipeline.exam_paths.exam_artifact_dir`) and reused
if no source PDF is newer than the cache. Exam PDF figures go under
``{artifact_dir}/scaffold_images/exam``.
"""

from __future__ import annotations

import json
import shutil
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
from pipeline.exam_paths import (
    artifact_overlays_dir,
    artifact_scaffold_cache_path,
    exam_artifact_dir,
    safe_path_stem,
)
from pipeline.pdf_parser import (
    merge_answers_into_scaffold,
    parse_answer_key_pdf,
    parse_exam_pdf,
    prepare_scaffold_image_dirs,
)
from pipeline.pdf_parser.content import normalize_multiple_choice_tree
from pipeline.scaffold_overlay import write_scaffold_boxes_pdf


SCHEMA_VERSION = 15


def _find_exam_pdf(folder: Path) -> Path:
    """Pick the vector exam PDF for parsing.

    When a four-up raw exam exists (see :func:`pipeline.bbox_projection.find_raw_four_up_pdf`),
    it is **always** used so question bboxes live in the same PDF space as IGCSE anchors used
    to project onto scans. Otherwise fall back to any raw/exam PDF (e.g. 2-up, 1-up).
    """
    from pipeline.bbox_projection import find_raw_four_up_pdf

    four_up = find_raw_four_up_pdf(folder)
    if four_up is not None:
        return four_up

    pdfs = [
        f
        for f in folder.glob("*.pdf")
        if ("raw" in f.name.lower() or "exam" in f.name.lower())
        and "answer" not in f.name.lower()
        and "scan" not in f.name.lower()
    ]
    if not pdfs:
        raise FileNotFoundError(f"No raw exam PDF found in {folder}")
    return sorted(pdfs, key=lambda p: p.name.lower())[0]


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

def _legacy_cache_path(folder: Path) -> Path:
    """Pre-layout: cache lived at the exam folder root."""
    return folder / "scaffold_cache.json"


def _legacy_scaffold_subdir_cache(folder: Path) -> Path:
    return folder / "scaffolds" / "scaffold_cache.json"


def _effective_cache_path(folder: Path, artifact_dir: Path) -> Path | None:
    for p in (
        artifact_scaffold_cache_path(artifact_dir),
        _legacy_scaffold_subdir_cache(folder),
        _legacy_cache_path(folder),
    ):
        if p.is_file():
            return p
    return None


def _source_pdfs(folder: Path) -> list[Path]:
    """Return only the PDFs that the scaffold is built from (exam + answer key).

    Derived outputs (cleaned_scan.pdf, overlay PDFs, etc.) are intentionally excluded
    so they don't invalidate the cache when they are updated after a deskew run.
    """
    try:
        exam = _find_exam_pdf(folder)
        sources = [exam]
    except FileNotFoundError:
        sources = []
    ans = _find_answer_pdf(folder)
    if ans is not None:
        sources.append(ans)
    return sources


def _is_cache_valid(folder: Path, artifact_dir: Path) -> bool:
    cache = _effective_cache_path(folder, artifact_dir)
    if cache is None:
        return False
    sources = _source_pdfs(folder)
    if not sources:
        return False
    cache_mtime = cache.stat().st_mtime
    for pdf in sources:
        if pdf.stat().st_mtime > cache_mtime:
            return False
    return True


def _cache_path_under_exam_folder(path: Path, exam_folder: Path) -> bool:
    try:
        path.resolve().relative_to(exam_folder.resolve())
        return True
    except ValueError:
        return False


def _migrate_scaffold_cache_to_artifact(
    exam_folder: Path, artifact_dir: Path, scaffold: ExamScaffold
) -> None:
    """Copy scaffold JSON + images into *artifact_dir* and remove legacy copies in *exam_folder*."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "scaffolds").mkdir(parents=True, exist_ok=True)
    _save_cache(artifact_dir, scaffold)
    src_img = exam_folder / "scaffold_images"
    dst_img = artifact_dir / "scaffold_images"
    if src_img.is_dir():
        if dst_img.exists():
            shutil.rmtree(dst_img)
        shutil.copytree(src_img, dst_img)
    _clear_legacy_scaffold_outputs(exam_folder)


def _clear_legacy_scaffold_outputs(exam_folder: Path) -> None:
    for p in (_legacy_scaffold_subdir_cache(exam_folder), _legacy_cache_path(exam_folder)):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    leg_img = exam_folder / "scaffold_images"
    if leg_img.is_dir():
        shutil.rmtree(leg_img, ignore_errors=True)
    leg_sd = exam_folder / "scaffolds"
    if leg_sd.is_dir():
        try:
            if not any(leg_sd.iterdir()):
                leg_sd.rmdir()
        except OSError:
            pass


def _load_cache(folder: Path, artifact_dir: Path) -> ExamScaffold:
    path = _effective_cache_path(folder, artifact_dir)
    if path is None:
        raise FileNotFoundError(f"No scaffold cache for {folder}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "scaffold cache schema_version mismatch — rebuild required "
            f"(got {data.get('schema_version')!r}, need {SCHEMA_VERSION})"
        )
    questions = [question_from_dict(q) for q in data["questions"]]
    total = int(data.get("total_marks", 0))
    if not total and questions:
        total = sum(q.marks for q in gradable_questions(questions))
    return ExamScaffold(
        questions=questions,
        total_marks=total,
        page_count=int(data.get("page_count", 0)),
        raw_description=data.get("raw_description", ""),
    )


def _save_cache(artifact_dir: Path, scaffold: ExamScaffold) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "questions": [question_to_dict(q) for q in scaffold.questions],
        "total_marks": scaffold.total_marks,
        "page_count": scaffold.page_count,
        "raw_description": scaffold.raw_description,
    }
    out = artifact_scaffold_cache_path(artifact_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_scaffold(
    folder: Path,
    client: Any | None = None,
    dpi: int = 200,
    *,
    artifact_dir: Path | None = None,
    output_base: str | Path = "output",
) -> ExamScaffold:
    """Build (or load from cache) the ExamScaffold for the exam in *folder*.

    Derived files (cache, ``scaffold_images``, overlay PDF) go under *artifact_dir*
    (default: ``output/<exam_stem>/``). *client* is optional and unused.
    *dpi* is unused; parsing is vector-based.
    """
    _ = client, dpi
    from pipeline.terminal_ui import tool_line

    ad = artifact_dir or exam_artifact_dir(folder, output_base)

    if _is_cache_valid(folder, ad):
        try:
            tool_line("scaffold", "Loading scaffold from cache …")
            loaded_path = _effective_cache_path(folder, ad)
            scaffold = _load_cache(folder, ad)
            if loaded_path is not None and _cache_path_under_exam_folder(
                loaded_path, folder
            ):
                tool_line(
                    "scaffold",
                    "Migrating scaffold cache and images from exam folder → output …",
                )
                _migrate_scaffold_cache_to_artifact(folder, ad, scaffold)
            return scaffold
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            tool_line("scaffold", "Cache incompatible or corrupt — rebuilding …")

    exam_pdf = _find_exam_pdf(folder)
    prepare_scaffold_image_dirs(ad)

    tool_line("scaffold", f"Parsing exam PDF (vector): {exam_pdf.name} …")
    questions = parse_exam_pdf(exam_pdf, folder, artifact_dir=ad)
    if not questions:
        raise RuntimeError(
            "No questions detected in exam PDF. Check that the file is a vector paper "
            "with Cambridge-style left-margin question numbers."
        )

    ans = _find_answer_pdf(folder)
    if ans is not None:
        tool_line("scaffold", f"Parsing answer key PDF (vector): {ans.name} …")
        amap, table_answers, printed_mc = parse_answer_key_pdf(ans, folder)
        merge_answers_into_scaffold(
            questions,
            amap,
            table_model_answers=table_answers,
            printed_mc_letters=printed_mc,
        )
    else:
        tool_line("scaffold", "No answer key PDF found — correct_answer left empty.")

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
    _save_cache(ad, scaffold)
    _clear_legacy_scaffold_outputs(folder)
    od = artifact_overlays_dir(ad)
    od.mkdir(parents=True, exist_ok=True)
    boxes_out = od / f"{safe_path_stem(exam_pdf.stem)}_scaffold_boxes.pdf"
    out_pdf, n_rects, n_pages = write_scaffold_boxes_pdf(
        exam_pdf, questions, output_path=boxes_out
    )
    tool_line(
        "scaffold",
        f"Scaffold built: {len(questions)} top-level questions, "
        f"{len(leaves)} gradable parts, {total_marks} total marks.",
    )
    tool_line(
        "scaffold",
        f"Bounding-box overlay: {out_pdf.name} ({n_rects} rectangles on {n_pages} page(s)).",
    )
    return scaffold
