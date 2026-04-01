"""Build an ExamScaffold by AI-analysing the raw exam PDF and the answer key.

Step B: analyse the raw exam paper → list of Question objects (no answers yet).
Step C: analyse the answer key PDF → fill in correct_answer and marking_criteria.

Results are cached as ``{folder}/scaffold_cache.json`` and reused if the cache
is newer than both source PDFs.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from pipeline.models import ExamScaffold, Question


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXAM_STRUCTURE_PROMPT = """\
You are analysing a scanned exam paper page. Your task is to identify ALL questions \
visible on this page.

For each question return a JSON object:
{
  "questions": [
    {
      "number": "1",
      "question_type": "multiple_choice" | "short_answer" | "calculation" | "long_answer",
      "content_summary": "Brief description of what the question asks",
      "marks": 2
    }
  ]
}

Rules:
- "number" must be a string that uniquely identifies the question (e.g. "1", "2a", "38").
- "marks" is the maximum mark available; use 1 if not shown.
- If no questions are visible (e.g. cover page or blank), return {"questions": []}.
Return ONLY the JSON object.
"""

_ANSWER_KEY_PROMPT = """\
You are analysing an exam answer key page. Extract the correct answer and any \
marking criteria for each question.

Return a JSON object:
{
  "answers": [
    {
      "number": "1",
      "correct_answer": "B",
      "marking_criteria": "Accept any answer showing ... (optional)"
    }
  ]
}

Rules:
- "number" must match the question numbering in the original exam.
- For multiple-choice, "correct_answer" is the letter (A/B/C/D).
- For written questions, give a concise model answer or key points.
- If no answer information is visible, return {"answers": []}.
Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pages_to_b64_jpegs(pdf_path: Path, dpi: int = 200) -> list[str]:
    """Render a PDF to JPEG images and return base64-encoded strings."""
    from pdf2image import convert_from_path

    pages = convert_from_path(str(pdf_path), dpi=dpi, thread_count=os.cpu_count() or 4)
    result = []
    for page in pages:
        buf = io.BytesIO()
        page.convert("RGB").save(buf, format="JPEG", quality=85)
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def _kimi_vision_call(client: Any, image_b64: str, prompt: str, max_tokens: int = 2048) -> str:
    """Single Kimi vision call; returns raw response text."""
    model = os.getenv("PIPELINE_AI_MODEL") or "kimi-k2.5"
    is_k2_5 = model.startswith("kimi-k2")
    extra: dict = {}
    if is_k2_5:
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                **extra,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [scaffold] API error (attempt {attempt}/3): {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return ""


def _parse_json_safe(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Step B: analyse exam structure
# ---------------------------------------------------------------------------

def _analyze_exam_structure(folder: Path, client: Any, dpi: int = 200) -> list[Question]:
    """Return Questions extracted from the raw exam PDF (no answers yet)."""
    pdfs = [f for f in folder.glob("*.pdf")
            if "raw" in f.name.lower() or "exam" in f.name.lower()
            if "answer" not in f.name.lower() and "scan" not in f.name.lower()]
    if not pdfs:
        raise FileNotFoundError(f"No raw exam PDF found in {folder}")
    exam_pdf = pdfs[0]

    print(f"[scaffold] Analysing exam structure: {exam_pdf.name} …")
    pages_b64 = _pages_to_b64_jpegs(exam_pdf, dpi=dpi)

    seen_numbers: set[str] = set()
    questions: list[Question] = []

    for i, img_b64 in enumerate(pages_b64, 1):
        print(f"  Page {i}/{len(pages_b64)} …", end=" ", flush=True)
        raw = _kimi_vision_call(client, img_b64, _EXAM_STRUCTURE_PROMPT)
        data = _parse_json_safe(raw)
        page_qs = data.get("questions", [])
        added = 0
        for q in page_qs:
            num = str(q.get("number", "")).strip()
            if not num or num in seen_numbers:
                continue
            seen_numbers.add(num)
            questions.append(Question(
                number=num,
                question_type=q.get("question_type", "short_answer"),
                content_summary=q.get("content_summary", ""),
                marks=int(q.get("marks", 1)),
            ))
            added += 1
        print(f"{added} question(s) found")

    return questions


# ---------------------------------------------------------------------------
# Step C: extract answer key
# ---------------------------------------------------------------------------

def _extract_answer_key(folder: Path, client: Any, questions: list[Question], dpi: int = 200) -> list[Question]:
    """Merge correct answers and marking criteria into *questions* from the answer key PDF."""
    answer_pdfs = [f for f in folder.glob("*.pdf") if "answer" in f.name.lower()]
    if not answer_pdfs:
        print("[scaffold] No answer key PDF found — answers will be left blank.")
        return questions
    answer_pdf = answer_pdfs[0]

    print(f"[scaffold] Extracting answer key: {answer_pdf.name} …")
    pages_b64 = _pages_to_b64_jpegs(answer_pdf, dpi=dpi)

    answer_map: dict[str, dict] = {}
    for i, img_b64 in enumerate(pages_b64, 1):
        print(f"  Page {i}/{len(pages_b64)} …", end=" ", flush=True)
        raw = _kimi_vision_call(client, img_b64, _ANSWER_KEY_PROMPT)
        data = _parse_json_safe(raw)
        answers = data.get("answers", [])
        print(f"{len(answers)} answer(s) found")
        for a in answers:
            num = str(a.get("number", "")).strip()
            if num:
                answer_map[num] = a

    for q in questions:
        if q.number in answer_map:
            entry = answer_map[q.number]
            q.correct_answer = entry.get("correct_answer") or q.correct_answer
            q.marking_criteria = entry.get("marking_criteria") or q.marking_criteria

    return questions


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(folder: Path) -> Path:
    return folder / "scaffold_cache.json"


def _is_cache_valid(folder: Path) -> bool:
    cache = _cache_path(folder)
    if not cache.exists():
        return False
    cache_mtime = cache.stat().st_mtime
    for pdf in folder.glob("*.pdf"):
        if pdf.stat().st_mtime > cache_mtime:
            return False
    return True


def _load_cache(folder: Path) -> ExamScaffold:
    with open(_cache_path(folder), encoding="utf-8") as f:
        data = json.load(f)
    questions = [Question(**q) for q in data["questions"]]
    return ExamScaffold(
        questions=questions,
        total_marks=data["total_marks"],
        raw_description=data.get("raw_description", ""),
    )


def _save_cache(folder: Path, scaffold: ExamScaffold) -> None:
    data = {
        "questions": [q.__dict__ for q in scaffold.questions],
        "total_marks": scaffold.total_marks,
        "raw_description": scaffold.raw_description,
    }
    with open(_cache_path(folder), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_scaffold(folder: Path, client: Any, dpi: int = 200) -> ExamScaffold:
    """Build (or load from cache) the ExamScaffold for the exam in *folder*."""
    if _is_cache_valid(folder):
        print("[scaffold] Loading scaffold from cache …")
        return _load_cache(folder)

    questions = _analyze_exam_structure(folder, client, dpi=dpi)
    questions = _extract_answer_key(folder, client, questions, dpi=dpi)

    total_marks = sum(q.marks for q in questions)
    raw_description = (
        f"{len(questions)} questions totalling {total_marks} marks; "
        + ", ".join(f"Q{q.number}({q.marks}m)" for q in questions)
    )

    scaffold = ExamScaffold(
        questions=questions,
        total_marks=total_marks,
        raw_description=raw_description,
    )
    _save_cache(folder, scaffold)
    print(f"[scaffold] Scaffold built: {len(questions)} questions, {total_marks} total marks.")
    return scaffold
