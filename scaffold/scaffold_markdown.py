"""Write a human-readable ``scaffold.md`` from the same dict payload as ``scaffold.json``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from shared.exam_paths import artifact_scaffold_markdown_path


def _heading_prefix(depth: int) -> str:
    level = min(3 + depth, 6)
    return "#" * level


def _type_label(question_type: str) -> str:
    return str(question_type).replace("_", " ").strip() or "question"


def _marks_word(marks: int) -> str:
    n = int(marks)
    return "1 mark" if n == 1 else f"{n} marks"


def _strip_fill_in_dots(text: str) -> str:
    """Remove trailing lines that are only dot runs (Cambridge answer lines)."""
    lines = text.split("\n")
    while lines and re.match(r"^\s*\.+\s*$", lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _escape_table_cell(s: str) -> str:
    return s.replace("|", "\\|")


def _format_prose_block(label: str, value: str) -> list[str]:
    out: list[str] = []
    if "\n" in value.strip():
        out.append(f"**{label}:**")
        out.append("")
        for line in value.split("\n"):
            out.append(f"> {line}")
    else:
        out.append(f"**{label}:** {value}")
    return out


def _render_question(q: dict[str, Any], depth: int, lines: list[str]) -> None:
    num = q.get("number", "")
    qtype = _type_label(str(q.get("question_type", "")))
    marks = int(q.get("marks", 0))
    hp = _heading_prefix(depth)
    lines.append(f"{hp} Q{num} · {qtype} · {_marks_word(marks)}")
    lines.append("")

    text = q.get("text")
    if isinstance(text, str) and text.strip():
        lines.append(_strip_fill_in_dots(text))
        lines.append("")

    opts = q.get("answer_options")
    if isinstance(opts, list) and opts:
        lines.append("**Options**")
        lines.append("")
        lines.append("| Letter | Text |")
        lines.append("|--------|------|")
        for o in opts:
            if not isinstance(o, dict):
                continue
            letter = _escape_table_cell(str(o.get("letter", "")))
            ot = _escape_table_cell(str(o.get("text") or ""))
            lines.append(f"| {letter} | {ot} |")
        lines.append("")

    for img_key in ("images", "answer_images"):
        imgs = q.get(img_key)
        if not isinstance(imgs, list) or not imgs:
            continue
        label = "Image" if img_key == "images" else "Answer image"
        for im in imgs:
            if isinstance(im, dict) and im.get("path"):
                lines.append(f"**{label}:** `{im['path']}`")
                lines.append("")

    ca = q.get("correct_answer")
    if isinstance(ca, str) and ca.strip():
        lines.extend(_format_prose_block("Answer", ca.strip()))
        lines.append("")

    mc = q.get("marking_criteria")
    if isinstance(mc, str) and mc.strip():
        lines.extend(_format_prose_block("Marking criteria", mc.strip()))
        lines.append("")

    subs = q.get("subquestions")
    if isinstance(subs, list) and subs:
        for s in subs:
            if isinstance(s, dict):
                _render_question(s, depth + 1, lines)


def write_scaffold_markdown(artifact_dir: Path, payload: dict[str, Any]) -> None:
    """Write ``scaffold.md`` next to ``scaffold.json`` (same folder as *artifact_dir*)."""
    lines: list[str] = []
    lines.append("# Exam Scaffold")
    lines.append("")
    sv = payload.get("schema_version", "")
    tm = payload.get("total_marks", "")
    pc = payload.get("page_count", "")
    lines.append(f"**Schema version:** {sv} · **Total marks:** {tm} · **Pages:** {pc}")
    lines.append("")

    raw = payload.get("raw_description")
    if isinstance(raw, str) and raw.strip():
        lines.append("## Summary")
        lines.append("")
        lines.append(raw.strip())
        lines.append("")

    questions = payload.get("questions")
    if isinstance(questions, list) and questions:
        lines.append("## Questions")
        lines.append("")
        first = True
        for q in questions:
            if not isinstance(q, dict):
                continue
            if not first:
                lines.append("---")
                lines.append("")
            first = False
            _render_question(q, 0, lines)

    path = artifact_scaffold_markdown_path(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines).rstrip() + "\n"
    path.write_text(body, encoding="utf-8")
