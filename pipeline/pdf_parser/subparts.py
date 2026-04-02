"""Split written stems into nested letter / roman sub-questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz

from pipeline.models import BBox, ExamImage, Question, WritingArea
from pipeline.pdf_parser.config import ParserConfig
from pipeline.pdf_parser.content import infer_marks, infer_question_type, rollup_question_marks
from pipeline.pdf_parser.layout import cell_margin_band, cell_scales, expand_bbox_to_subpage_width
from pipeline.pdf_parser.regions import clip_for_text_segment

_ROMAN_SUB_OK = frozenset(
    "i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx".split()
)

_RE_SUB_LETTER = re.compile(r"^\s*\(?([a-z])\)\s*(.*)$", re.I | re.DOTALL)
_RE_SUB_ROMAN = re.compile(r"^\s*\(([ivxlcdm]+)\)\s*(.*)$", re.I | re.DOTALL)
_ROOT_MARGIN_DIGITS = re.compile(r"^(\d{1,2})")


def root_margin_digits(question_number: str) -> str:
    """Leading main question digits for ``N (a) …`` matching (e.g. ``38`` from ``38_2``)."""
    m = _ROOT_MARGIN_DIGITS.match(question_number.strip())
    return m.group(1) if m else ""


@dataclass
class ClipLine:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class SubAnchor:
    kind: str
    label: str
    rest: str
    line_idx: int
    page: int
    y0: float
    y1: float
    x0: float
    x1: float


def _roman_sub_label(tok: str) -> str | None:
    t = tok.strip().lower()
    if t in _ROMAN_SUB_OK:
        return t
    return None


def _parse_subpart_anchor(line_text: str) -> tuple[str, str, str] | None:
    m = _RE_SUB_ROMAN.match(line_text)
    if m:
        lab = _roman_sub_label(m.group(1))
        if lab:
            return ("roman", lab, m.group(2).strip())
    m = _RE_SUB_LETTER.match(line_text)
    if m:
        ch = m.group(1).lower()
        if "a" <= ch <= "z":
            return ("letter", ch, m.group(2).strip())
    return None


def _parse_margin_plus_letter_line(line_text: str, margin_digits: str) -> tuple[str, str, str] | None:
    if not margin_digits or not margin_digits.isdigit():
        return None
    m = re.match(
        rf"^\s*{re.escape(margin_digits)}\s*\(?([a-z])\)\s*(.*)$",
        line_text,
        re.I | re.DOTALL,
    )
    if m and "a" <= m.group(1).lower() <= "z":
        return ("letter", m.group(1).lower(), m.group(2).strip())
    return None


def _merge_margin_digit_with_next_letter_line(
    lines: list[ClipLine], margin_digits: str
) -> list[ClipLine]:
    if not lines or not margin_digits.isdigit():
        return lines
    out: list[ClipLine] = []
    i = 0
    pn_pat = re.compile(rf"^\s*{re.escape(margin_digits)}\s*$")
    while i < len(lines):
        if i + 1 < len(lines) and pn_pat.match(lines[i].text):
            nxt = _parse_subpart_anchor(lines[i + 1].text)
            if nxt and nxt[0] == "letter":
                t = f"{lines[i].text.strip()} {lines[i + 1].text.strip()}"
                out.append(
                    ClipLine(
                        page=lines[i].page,
                        x0=min(lines[i].x0, lines[i + 1].x0),
                        y0=min(lines[i].y0, lines[i + 1].y0),
                        x1=max(lines[i].x1, lines[i + 1].x1),
                        y1=max(lines[i].y1, lines[i + 1].y1),
                        text=t,
                    )
                )
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return out


def _collect_clip_lines(page: fitz.Page, clip: fitz.Rect, page_1based: int) -> list[ClipLine]:
    out: list[ClipLine] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not line["spans"]:
                continue
            bb = line["bbox"]
            r = fitz.Rect(bb)
            if not r.intersects(clip):
                continue
            ir = r & clip
            if ir.get_area() < r.get_area() * 0.25:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            out.append(
                ClipLine(
                    page=page_1based,
                    x0=float(bb[0]),
                    y0=float(bb[1]),
                    x1=float(bb[2]),
                    y1=float(bb[3]),
                    text=text,
                )
            )
    out.sort(key=lambda ln: (ln.page, ln.y0, ln.x0))
    return out


def _parse_total_marks_footer(text: str) -> int | None:
    m = re.search(r"\[Total:\s*(\d+)\s*\]", text, re.I)
    return int(m.group(1)) if m else None


def _bbox_union_lines(lines: list[ClipLine], page: int) -> BBox | None:
    xs = [ln for ln in lines if ln.page == page]
    if not xs:
        return None
    return BBox(
        min(ln.x0 for ln in xs),
        min(ln.y0 for ln in xs),
        max(ln.x1 for ln in xs),
        max(ln.y1 for ln in xs),
        page,
    )


def _parent_letter_anchor_index(anchors: list[SubAnchor], roman_idx: int) -> int | None:
    for k in range(roman_idx - 1, -1, -1):
        if anchors[k].kind == "letter":
            return k
    return None


def _assign_asset_to_deepest_node(root: Question, wa: WritingArea) -> None:
    cy = (wa.bbox.y0 + wa.bbox.y1) * 0.5
    p = wa.bbox.page
    best: Question | None = None
    best_depth = -1

    def walk(q: Question, depth: int) -> None:
        nonlocal best, best_depth
        if q.bbox.page != p:
            return
        if q.bbox.y0 - 1 <= cy <= q.bbox.y1 + 1:
            if depth > best_depth:
                best_depth = depth
                best = q
        for s in q.subquestions:
            walk(s, depth + 1)

    walk(root, 0)
    (best or root).writing_areas.append(wa)


def _assign_image_to_deepest_node(root: Question, im: ExamImage) -> None:
    cy = (im.bbox.y0 + im.bbox.y1) * 0.5
    p = im.bbox.page
    best: Question | None = None
    best_depth = -1

    def walk(q: Question, depth: int) -> None:
        nonlocal best, best_depth
        if q.bbox.page != p:
            return
        if q.bbox.y0 - 1 <= cy <= q.bbox.y1 + 1:
            if depth > best_depth:
                best_depth = depth
                best = q
        for s in q.subquestions:
            walk(s, depth + 1)

    walk(root, 0)
    (best or root).images.append(im)


def maybe_split_written_subquestions(
    q: Question,
    doc: fitz.Document,
    segs: list[tuple[int, float, float, fitz.Rect, int, float, bool, bool]],
    cfg: ParserConfig,
) -> Question:
    if q.question_type == "multiple_choice":
        return q

    margin_digits = root_margin_digits(q.number)
    lines: list[ClipLine] = []
    for pidx, y0, y1, cell, _pr, num_x1, _st, _sb in segs:
        mt, _mb = cell_margin_band(cell, cfg)
        _sx, sy = cell_scales(cell)
        pad_above = min(cfg.padding_above * sy, cfg.text_clip_pad_above_pt * sy)
        text_y0 = max(y0 - pad_above, mt)
        tc = clip_for_text_segment(doc, pidx, text_y0, y1, cfg, cell, num_x1)
        lines.extend(_collect_clip_lines(doc[pidx], tc, pidx + 1))

    if len(lines) < 3:
        return q

    lines = _merge_margin_digit_with_next_letter_line(lines, margin_digits)

    anchors: list[SubAnchor] = []
    for i, ln in enumerate(lines):
        parsed = _parse_margin_plus_letter_line(ln.text, margin_digits) or _parse_subpart_anchor(
            ln.text
        )
        if not parsed:
            continue
        kind, label, rest = parsed
        anchors.append(
            SubAnchor(
                kind=kind,
                label=label,
                rest=rest,
                line_idx=i,
                page=ln.page,
                y0=ln.y0,
                y1=ln.y1,
                x0=ln.x0,
                x1=ln.x1,
            )
        )

    if len(anchors) < 2:
        return q

    n = len(anchors)

    def slice_lines(ai: int) -> tuple[list[ClipLine], str]:
        a = anchors[ai]
        start_i = a.line_idx
        end_line_exclusive = anchors[ai + 1].line_idx if ai + 1 < n else len(lines)
        seg_lines = lines[start_i:end_line_exclusive]
        if not seg_lines:
            return [], ""
        parts: list[str] = []
        if a.rest:
            parts.append(a.rest)
        parts.extend(ln.text for ln in seg_lines[1:])
        return seg_lines, "\n".join(parts).strip()

    def _segment_y1_for_page(page_1based: int) -> float:
        """Bottom y of the question's region on *page_1based* (from the raw segments)."""
        for pidx, y0, y1, cell, _pr, _nx1, _st, _sb in segs:
            if pidx + 1 == page_1based:
                return float(y1)
        return q.bbox.y1

    def _extend_subpart_bbox_to_next_anchor(bb: BBox, next_a: SubAnchor | None) -> BBox:
        """Stretch *bb* so its bottom reaches the top of *next_a* (same page) or the
        segment bottom (different page).  Only ever extends, never shrinks."""
        if next_a is None:
            return bb
        if next_a.page == bb.page:
            new_y1 = max(bb.y1, next_a.y0)
        else:
            new_y1 = max(bb.y1, _segment_y1_for_page(bb.page))
        if new_y1 <= bb.y1:
            return bb
        return BBox(bb.x0, bb.y0, bb.x1, new_y1, bb.page)

    anchor_nodes: dict[int, Question] = {}
    root_children: list[Question] = []

    for ai, a in enumerate(anchors):
        seg_lines, seg_text = slice_lines(ai)
        if not seg_text:
            continue
        page = seg_lines[0].page
        bb = _bbox_union_lines(seg_lines, page) or q.bbox
        next_a = anchors[ai + 1] if ai + 1 < n else None
        bb = _extend_subpart_bbox_to_next_anchor(bb, next_a)
        bb = expand_bbox_to_subpage_width(doc, bb)
        sq = Question(
            number="",
            question_type=infer_question_type(seg_text),
            text=seg_text,
            marks=infer_marks(seg_text),
            bbox=bb,
            images=[],
            writing_areas=[],
            subquestions=[],
        )
        anchor_nodes[ai] = sq

    for ai, a in enumerate(anchors):
        node = anchor_nodes.get(ai)
        if node is None:
            continue
        if a.kind == "letter":
            node.number = f"{q.number}{a.label}"
            root_children.append(node)
        else:
            pli = _parent_letter_anchor_index(anchors, ai)
            if pli is not None and pli in anchor_nodes:
                parent_sq = anchor_nodes[pli]
                node.number = f"{parent_sq.number}{a.label}"
                parent_sq.subquestions.append(node)
            else:
                node.number = f"{q.number}{a.label}"
                root_children.append(node)

    if not root_children:
        return q

    preamble_lines = lines[: anchors[0].line_idx]
    preamble = "\n".join(x.text for x in preamble_lines).strip()

    full_before = q.text
    total_m = _parse_total_marks_footer(full_before)
    q.text = preamble
    q.subquestions = root_children
    rollup_question_marks(q)
    if total_m is not None:
        q.marks = total_m

    areas = list(q.writing_areas)
    imgs = list(q.images)
    q.writing_areas = []
    q.images = []
    for wa in areas:
        _assign_asset_to_deepest_node(q, wa)
    for im in imgs:
        _assign_image_to_deepest_node(q, im)

    return q
