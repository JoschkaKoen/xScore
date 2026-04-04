"""Stem heuristics, drawings, and embedded images."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from shared.models import BBox, ExamImage, McAnswerOption, Question, WritingArea, flatten_questions
from scaffold.pdf_parser.config import DEFAULT_PARSER_CONFIG, ParserConfig

# Printed figure captions (exclude from vector-figure raster union).
_RE_FIG_CAPTION_LINE = re.compile(r"^Fig(?:\.|ure)?\b", re.I)
# Cambridge mini-page banner lines (e.g. "---IGCSE Physics: s24 23---") — not exercise content.
_RE_EXAM_SHEET_LABEL_LINE = re.compile(r"IGCSE\s+Physics", re.I)

# PDF often maps Symbol/Wingdings glyphs into the Unicode private-use area (PUA).
# These code points are font-specific; normalize to plain Unicode/ASCII for parsing.
_PDF_PUA_MULTIPLY = "\uf0b4"  # multiplication cross (displays as a private-use glyph)
_PDF_PUA_ARROW_RIGHT = "\uf0ae"
_PDF_PUA_ARROW_RIGHT_ALT = "\uf0ee"  # alternate PUA encoding for small right arrow
_PDF_PUA_BULLET = "\uf0b7"  # round bullet in mark-scheme / list lines


def normalize_pdf_multiplication_glyph(text: str) -> str:
    """Replace the common PDF multiplication PUA character (U+F0B4) with ASCII ``x``."""
    if not text:
        return text
    return text.replace(_PDF_PUA_MULTIPLY, "x")


def normalize_pdf_arrow_glyph(text: str) -> str:
    """Replace PDF PUA arrow glyphs with Unicode ``→`` (U+2192)."""
    if not text:
        return text
    t = text.replace(_PDF_PUA_ARROW_RIGHT, "→")
    return t.replace(_PDF_PUA_ARROW_RIGHT_ALT, "→")


def normalize_pdf_bullet_glyph(text: str) -> str:
    """Replace PDF bullet PUA (U+F0B7) with Unicode bullet U+2022."""
    if not text:
        return text
    return text.replace(_PDF_PUA_BULLET, "\u2022")


# En dash, hyphen-minus, minus sign — used before negative exponents in extracted PDF text.
_RE_SCI_EXPONENT_DASH = r"[–\-−\u2212]"


def normalize_scientific_powers_of_ten(text: str) -> str:
    """Turn ``x 10…`` / ``x 10–n`` (after ``x`` is normalized) into ``x 10^…``.

    Cambridge-style papers often omit the caret: ``1.5 x 1011`` → ``1.5 x 10^11``,
    ``2.3 x 10–18`` → ``2.3 x 10^-18``.
    """
    if not text:
        return text
    t = text
    t = re.sub(
        rf"x\s*10\s*{_RE_SCI_EXPONENT_DASH}\s*(\d+)",
        r"x 10^-\1",
        t,
    )
    t = re.sub(r"x\s*10\s*(\d+)(?=\D|$)", r"x 10^\1", t)
    # Mark schemes use a literal exponent letter (e.g. ``10N`` for ``10^N``); do not touch ``10^…``.
    t = re.sub(
        r"x\s*10(?!\^)\s*([A-Za-z])(?=$|\s|OR\b|[.,;:])",
        r"x 10^\1",
        t,
    )
    return t


def normalize_exam_scientific_text(text: str) -> str:
    """PUA multiply → ``x``, arrow → ``→``, bullet → ``•``, Unicode ``×`` → ``x``, then ``10^n`` fix."""
    if not text:
        return text
    t = normalize_pdf_multiplication_glyph(text)
    t = normalize_pdf_arrow_glyph(t)
    t = normalize_pdf_bullet_glyph(t)
    t = t.replace("×", "x")
    return normalize_scientific_powers_of_ten(t)



def _padded_crop_rect(page: fitz.Page, core: fitz.Rect, clip: fitz.Rect, pad: float) -> fitz.Rect:
    """Expand *core* by *pad* on all sides, clamped to page and intersected with *clip*."""
    pr = page.rect
    expanded = fitz.Rect(
        max(pr.x0, core.x0 - pad),
        max(pr.y0, core.y0 - pad),
        min(pr.x1, core.x1 + pad),
        min(pr.y1, core.y1 + pad),
    )
    return expanded & clip


def _vector_figure_rects_in_clip(page: fitz.Page, clip: fitz.Rect, cfg: ParserConfig) -> list[fitz.Rect]:
    """Drawing bboxes that look like compact figures (not margin rules or answer lines)."""
    out: list[fitz.Rect] = []
    for d in page.get_drawings():
        r = fitz.Rect(d.get("rect", (0, 0, 0, 0)))
        if r.is_empty or not r.intersects(clip):
            continue
        ir = r & clip
        w, h = ir.width, ir.height
        if w < 1.0 or h < 1.0:
            continue
        if min(w, h) < cfg.vector_figure_min_short_side:
            continue
        area = ir.get_area()
        if area < cfg.vector_figure_min_area:
            continue
        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / max(short_side, 0.01)
        if aspect > cfg.vector_figure_max_aspect:
            continue
        out.append(ir)
    return out


def _vector_figure_core_with_nearby_labels(
    page: fitz.Page, clip: fitz.Rect, figure: fitz.Rect, cfg: ParserConfig
) -> fitz.Rect:
    """Expand vector *figure* bbox to include short text lines in a padded band (e.g. A, B, Sun).

    Omits lines that look like figure captions (``Fig. 10.1``) so they are not in the crop.
    """
    hx = cfg.vector_figure_label_h_pad_pt
    vy0 = cfg.vector_figure_label_v_pad_top_pt
    vy1 = cfg.vector_figure_label_v_pad_bottom_pt
    search = fitz.Rect(
        max(clip.x0, figure.x0 - hx),
        max(clip.y0, figure.y0 - vy0),
        min(clip.x1, figure.x1 + hx),
        min(clip.y1, figure.y1 + vy1),
    ) & clip
    if search.is_empty:
        return figure
    core = figure
    max_chars = cfg.vector_figure_label_max_line_chars
    for block in page.get_text("dict", clip=clip)["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not line["spans"]:
                continue
            r = fitz.Rect(line["bbox"])
            if r.is_empty or not r.intersects(search):
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or len(text) > max_chars:
                continue
            if _RE_FIG_CAPTION_LINE.match(text):
                continue
            core |= r
    return core


def extract_images(
    page: fitz.Page,
    clip: fitz.Rect,
    artifact_dir: Path,
    stem: str,
    page_1based: int,
    img_counter: list[int],
    cfg: ParserConfig | None = None,
) -> list[ExamImage]:
    cfg = cfg or DEFAULT_PARSER_CONFIG
    out_dir = artifact_dir / "scaffold_images"
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
            crop = _padded_crop_rect(page, ir, clip, cfg.image_crop_pad_pt)
            if crop.is_empty:
                continue
            img_counter[0] += 1
            name = f"{stem}_{img_counter[0]}.png"
            abs_path = out_dir / name
            try:
                pix = page.get_pixmap(clip=crop, dpi=150)
                pix.save(str(abs_path))
            except Exception:
                img_counter[0] -= 1
                continue
            rel = f"scaffold_images/{name}"
            found.append(
                ExamImage(
                    bbox=BBox(crop.x0, crop.y0, crop.x1, crop.y1, page_1based),
                    path=rel,
                )
            )

    if not found and cfg.vector_figure_fallback:
        cands = _vector_figure_rects_in_clip(page, clip, cfg)
        if cands:
            best = max(cands, key=lambda z: z.get_area())
            core = _vector_figure_core_with_nearby_labels(page, clip, best, cfg)
            crop = _padded_crop_rect(page, core, clip, cfg.image_crop_pad_pt)
            if not crop.is_empty:
                img_counter[0] += 1
                name = f"{stem}_{img_counter[0]}.png"
                abs_path = out_dir / name
                try:
                    pix = page.get_pixmap(clip=crop, dpi=150)
                    pix.save(str(abs_path))
                except Exception:
                    img_counter[0] -= 1
                else:
                    rel = f"scaffold_images/{name}"
                    found.append(
                        ExamImage(
                            bbox=BBox(crop.x0, crop.y0, crop.x1, crop.y1, page_1based),
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
    """Apply :func:`strip_exam_mark_indicators`, multiply glyph cleanup, and ``10^n`` text fix."""
    q.text = normalize_exam_scientific_text(strip_exam_mark_indicators(q.text))
    for opt in q.answer_options:
        opt.text = normalize_exam_scientific_text(opt.text)
    for sq in q.subquestions:
        strip_question_tree_stems(sq)


def normalize_multiple_choice_tree(q: Question) -> None:
    """MC parts use the exercise region for ticks/circles — drop ruled boxes."""
    if q.question_type == "multiple_choice":
        q.writing_areas.clear()
        q.equation_blank_bboxes.clear()
    for sq in q.subquestions:
        normalize_multiple_choice_tree(sq)


def rollup_question_marks(q: Question) -> int:
    """Set each non-leaf ``marks`` to the sum of its descendants; leaves unchanged."""
    if not q.subquestions:
        return q.marks
    s = sum(rollup_question_marks(sq) for sq in q.subquestions)
    q.marks = s
    return s


def _mc_options_block_start_index(lines: list[str]) -> int | None:
    """Index of line with ``A`` (alone) when ``B`` and ``C`` appear as letter lines below."""
    for i, ln in enumerate(lines):
        if not re.fullmatch(r"\s*A\s*", ln, re.I):
            continue
        tail = lines[i + 1 :]
        has_b = any(re.fullmatch(r"\s*B\s*", x, re.I) for x in tail)
        has_c = any(re.fullmatch(r"\s*C\s*", x, re.I) for x in tail)
        if has_b and has_c:
            return i
    return None


def _is_mc_footer_line(ln: str) -> bool:
    t = ln.strip()
    if not t:
        return False
    if re.match(r"^IGCSE\s+Physics", t, re.I):
        return True
    if "permission to reproduce" in t.lower():
        return True
    return False


def _normalize_option_body(parts: list[str]) -> str:
    s = " ".join(x.strip() for x in parts if x.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_exam_scientific_text(s)


def split_mc_options_from_stem(raw: str) -> tuple[str, list[McAnswerOption]]:
    """Split Cambridge-style MC (letter on its own line) into stem and options.

    Returns ``(stem, options)``; if no recognizable block, ``(raw, [])``.
    """
    if not raw or not raw.strip():
        return raw, []
    lines = raw.splitlines()
    start = _mc_options_block_start_index(lines)
    if start is None:
        return raw, []
    stem = "\n".join(lines[:start]).strip()
    options: list[McAnswerOption] = []
    letter: str | None = None
    buf: list[str] = []
    for ln in lines[start:]:
        if _is_mc_footer_line(ln):
            break
        m = re.fullmatch(r"\s*([A-D])\s*", ln, re.I)
        if m:
            if letter is not None:
                body = _normalize_option_body(buf)
                if body:
                    options.append(McAnswerOption(letter=letter, text=body))
            letter = m.group(1).upper()
            buf = []
        else:
            buf.append(ln)
    if letter is not None:
        body = _normalize_option_body(buf)
        if body:
            options.append(McAnswerOption(letter=letter, text=body))
    if len(options) < 2:
        return raw, []
    stem = normalize_exam_scientific_text(stem)
    return stem, options


def mc_answer_options_display(options: list[McAnswerOption]) -> str:
    """``A: …  B: …`` style single line for scaffold / reports."""
    return "  ".join(f"{o.letter}: {o.text}" for o in options)


def ensure_multiple_choice_options_parsed(q: Question) -> None:
    """If *q* is MC but options were not split (e.g. type set from answer key), split *q.text*."""
    if q.question_type != "multiple_choice":
        for sq in q.subquestions:
            ensure_multiple_choice_options_parsed(sq)
        return
    if not q.answer_options:
        stem, opts = split_mc_options_from_stem(q.text)
        if opts:
            q.text = stem
            q.answer_options = opts
    for sq in q.subquestions:
        ensure_multiple_choice_options_parsed(sq)


def infer_question_type(text: str) -> str:
    tl = text.lower()
    if "multiple choice" in tl or re.search(r"(?m)^\s*[A-Da-d]\s{1,4}\d", text):
        return "multiple_choice"
    lines = text.splitlines()
    if _mc_options_block_start_index(lines) is not None:
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


def adjust_leaf_bboxes_after_previous_exercise(
    doc: fitz.Document,
    cfg: ParserConfig,
    questions: list[Question],
) -> None:
    """Pull each non-first leaf bbox.y0 down to sit just below the previous exercise's last line.

    Groups leaf questions by layout cell, sorts by y0, then for each leaf after the first
    in that cell scans the PDF text above it (skipping ``[Total: …]`` and Cambridge sheet
    label lines) to find the last printed line's bottom edge and sets bbox.y0 = that y1 + gap.

    After adjusting all bboxes, re-runs ``assign_answer_field_bboxes`` so
    ``equation_blank_bboxes`` reflect the updated regions.
    """
    from collections import defaultdict
    from scaffold.pdf_parser.answer_fields import assign_answer_field_bboxes
    from scaffold.pdf_parser.layout import cell_for_point
    from scaffold.pdf_parser.regions import clip_horizontal_bounds

    leaves: list[Question] = [q for q in flatten_questions(questions) if not q.subquestions]

    page_cache: dict[int, fitz.Page] = {}
    cell_groups: dict[tuple, list[Question]] = defaultdict(list)

    for q in leaves:
        pi = q.bbox.page - 1
        if pi < 0 or pi >= len(doc):
            continue
        if pi not in page_cache:
            page_cache[pi] = doc[pi]
        page = page_cache[pi]
        cx = (q.bbox.x0 + q.bbox.x1) * 0.5
        cy = (q.bbox.y0 + q.bbox.y1) * 0.5
        cell = cell_for_point(page, cx, cy)
        key = (pi, round(cell.x0), round(cell.y0), round(cell.x1), round(cell.y1))
        cell_groups[key].append(q)

    gap = cfg.leaf_bbox_gap_after_previous_line_pt

    for key, group in cell_groups.items():
        pi = key[0]
        page = page_cache[pi]
        group.sort(key=lambda q: (q.bbox.y0, q.bbox.x0))

        for i, q in enumerate(group):
            if i == 0:
                continue
            cx = (q.bbox.x0 + q.bbox.x1) * 0.5
            cy = (q.bbox.y0 + q.bbox.y1) * 0.5
            cell = cell_for_point(page, cx, cy)
            h0, h1 = clip_horizontal_bounds(doc, pi, cfg, cell)

            band = fitz.Rect(h0, float(cell.y0), h1, q.bbox.y0)
            last_y1 = float(cell.y0)
            for block in page.get_text("dict", clip=band)["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    bb = line["bbox"]
                    if bb[3] > q.bbox.y0:
                        continue
                    t = "".join(s["text"] for s in line["spans"]).strip()
                    if not t:
                        continue
                    if re.match(r"^\[Total:", t, re.I):
                        continue
                    if _RE_EXAM_SHEET_LABEL_LINE.search(t):
                        continue
                    if bb[3] > last_y1:
                        last_y1 = float(bb[3])

            new_y0 = last_y1 + gap
            if new_y0 < q.bbox.y0:
                q.bbox = BBox(q.bbox.x0, new_y0, q.bbox.x1, q.bbox.y1, q.bbox.page)

    for q in questions:
        assign_answer_field_bboxes(doc, cfg, q)
