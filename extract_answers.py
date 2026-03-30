#!/usr/bin/env python3
"""
extract_answers.py
------------------
Extracts student names + handwritten answers (Q38, Q39, Q40) from scanned IGCSE
answer sheets using Gemini Vision. Processes each page, crops the top half, and
saves results to JSON and a compiled PDF report.

Requirements:
    pip install google-genai pdf2image pillow python-dotenv pydantic
    brew install poppler   # macOS

    cd /Users/joschka/Desktop/Programming/Auto-Grader
    source .venv/bin/activate
    python extract_answers.py
    python extract_answers.py output/some_other.pdf
"""

import argparse
import io
import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from pydantic import BaseModel
from PIL import Image

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_PDF = "output/20260330135527722.pdf"
SAVE_DEBUG_IMAGES = True

PDF_DPI = 300  # 150 DPI is sufficient for Gemini to read handwritten letters and names
JPEG_QUALITY = 95
CROP_TOP_FRACTION = 0.5

GEMINI_MODEL = "gemini-3-flash-preview"
API_CALL_DELAY_S = 1.5
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5


# ---------------------------------------------------------------------------
# Structured output schema — enforced at the token level by Gemini
# ---------------------------------------------------------------------------

class StudentAnswers(BaseModel):
    student_name: str
    q38_left_top: str    # left side, position 1 (top)     — Q38
    q39_left: str        # left side, position 2           — Q39
    q40_left: str        # left side, position 3           — Q40
    q38_left_bottom: str # left side, position 4 (bottom)  — Q38 again
    q39_right: str       # right side, position 1 (top)    — Q39
    q40_right: str       # right side, position 2          — Q40
    confidence: str


PROMPT = """\
You are reading a scanned student exam answer sheet for a multiple-choice test.
Each student circles or writes a single letter — A, B, C, or D — for each question.

On the LEFT side of the page (in order from top to bottom):
  - Question 38 answer  → field: q38_left_top
  - Question 39 answer  → field: q39_left
  - Question 40 answer  → field: q40_left
  - Question 38 answer  → field: q38_left_bottom  (Q38 appears a second time at the bottom)

On the RIGHT side of the page (in order from top to bottom):
  - Question 39 answer  → field: q39_right
  - Question 40 answer  → field: q40_right

At the TOP of the page, the student has written their name in English.

Rules:
- For each question field, return ONLY the single letter the student wrote: A, B, C, or D.
- If a letter is illegible or missing, return "?" for that field.
- For student_name, return the exact name as written, or "UNKNOWN" if illegible.
- For confidence, return "high", "medium", or "low" based on overall legibility.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crop_top(image: Image.Image, fraction: float = CROP_TOP_FRACTION) -> Image.Image:
    """Return the top `fraction` of the image."""
    w, h = image.size
    return image.crop((0, 0, w, int(h * fraction)))


def to_jpeg_bytes(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    """Convert a PIL image to JPEG bytes."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def call_gemini(client: genai.Client, image_bytes: bytes, page_num: int) -> dict:
    """Call Gemini Vision with structured output + retry + exponential backoff.

    Config follows ``google.genai.types.GenerateContentConfig`` (see installed
    ``google-genai`` package). For models with *thinking* enabled, internal
    reasoning can consume ``max_output_tokens``; ``ThinkingConfig`` documents
    ``thinking_budget=0`` as DISABLED so the budget applies to the visible
    JSON response (avoids ``FinishReason.MAX_TOKENS`` on a few dozen chars).
    """
    last_error = None
    backoff = RETRY_BACKOFF_S

    # Typed config matches the API surface in types.GenerateContentConfig / TypedDict.
    gen_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_schema=StudentAnswers,
        thinking_config=types.ThinkingConfig(thinking_budget=-1),  # -1 = AUTOMATIC
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    PROMPT,
                ],
                config=gen_config,
            )
            # Log finish reason and full raw text to help diagnose truncation
            try:
                finish_reason = response.candidates[0].finish_reason
            except (IndexError, AttributeError):
                finish_reason = "unknown"
            if response.parsed:
                return response.parsed.model_dump()
            raw = response.text or ""
            print(f"\n    [DEBUG] finish_reason={finish_reason}")
            print(f"    [DEBUG] full response ({len(raw)} chars):\n{raw}\n")
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError) as parse_err:
                raise RuntimeError(
                    f"Unparseable response for page {page_num} "
                    f"(finish_reason={finish_reason})"
                ) from parse_err

        except Exception as e:
            print(f"    API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            last_error = e

        if attempt < MAX_RETRIES:
            print(f"    Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2

    return {
        "student_name": "EXTRACTION_ERROR",
        "q38_left_top": "",
        "q39_left": "",
        "q40_left": "",
        "q38_left_bottom": "",
        "q39_right": "",
        "q40_right": "",
        "confidence": "failed",
        "error": str(last_error),
    }


def load_existing_results(output_json: Path) -> dict[int, dict]:
    """Load existing JSON results so we can resume an interrupted run."""
    if not output_json.exists():
        return {}
    try:
        with open(output_json, encoding="utf-8") as f:
            records = json.load(f)
        return {r["page_number"]: r for r in records if "page_number" in r}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"WARNING: Could not parse existing results from {output_json} ({e}), starting fresh.")
        return {}


def save_results(results: list[dict], output_json: Path) -> None:
    """Persist current results to JSON."""
    sorted_results = sorted(results, key=lambda r: r.get("page_number", 0))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)


def _tex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def generate_report_pdf(results: list[dict], output_tex: Path, output_report: Path) -> None:
    """Generate a LaTeX table and compile to PDF."""
    sorted_results = sorted(results, key=lambda r: r.get("page_number", 0))

    rows = []
    for r in sorted_results:
        page_num = r.get("page_number", "?")
        name = _tex_escape(r.get("student_name", "UNKNOWN"))
        q38lt = _tex_escape(r.get("q38_left_top", "?"))
        q39l  = _tex_escape(r.get("q39_left", "?"))
        q40l  = _tex_escape(r.get("q40_left", "?"))
        q38lb = _tex_escape(r.get("q38_left_bottom", "?"))
        q39r  = _tex_escape(r.get("q39_right", "?"))
        q40r  = _tex_escape(r.get("q40_right", "?"))
        rows.append(f"        {page_num} & {name} & {q38lt} & {q39l} & {q40l} & {q38lb} & {q39r} & {q40r} \\\\")

    table_rows = "\n".join(rows)

    tex = f"""\
\\documentclass[a4paper,11pt]{{article}}
\\usepackage[margin=2cm]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{fontspec}}
\\usepackage{{xeCJK}}
\\setCJKmainfont{{PingFang SC Regular}}[BoldFont=PingFang SC Semibold]

\\title{{Student Answers Report}}
\\author{{Auto-Grader}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{longtable}}{{r l c c c c c c}}
    \\toprule
    \\textbf{{Page}} & \\textbf{{Student Name}} & \\textbf{{Q38 L↑}} & \\textbf{{Q39 L}} & \\textbf{{Q40 L}} & \\textbf{{Q38 L↓}} & \\textbf{{Q39 R}} & \\textbf{{Q40 R}} \\\\
    \\midrule
    \\endhead
{table_rows}
    \\bottomrule
\\end{{longtable}}

\\end{{document}}
"""

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"\nCompiling LaTeX -> {output_report} ...")
    result = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", str(output_tex)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  LaTeX compilation failed. Check {output_tex} for errors.")
        print(result.stdout[-500:] if result.stdout else "")
    else:
        print(f"  Report generated: {output_report}")

    # Clean up LaTeX auxiliary files
    for ext in (".aux", ".log", ".out"):
        aux = output_tex.with_suffix(ext)
        if aux.exists():
            aux.unlink()


def print_summary(results: list[dict]):
    """Print a quick summary of extraction quality."""
    total = len(results)
    high = sum(1 for r in results if r.get("confidence") == "high")
    medium = sum(1 for r in results if r.get("confidence") == "medium")
    low = sum(1 for r in results if r.get("confidence") == "low")
    failed = sum(1 for r in results if r.get("confidence") == "failed")
    unknown = sum(1 for r in results if r.get("student_name") in ("UNKNOWN", "EXTRACTION_ERROR"))

    print(f"\n{'=' * 50}")
    print(f"  SUMMARY: {total} pages processed")
    print(f"  High confidence:   {high}")
    print(f"  Medium confidence: {medium}")
    print(f"  Low confidence:    {low}")
    print(f"  Failed:            {failed}")
    print(f"  Unreadable names:  {unknown}")
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract handwritten answers from scanned exam PDFs using Gemini Vision."
    )
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF,
                        help=f"Path to input PDF (default: {DEFAULT_PDF})")
    parser.add_argument("--skip", action="store_true", default=False,
                        help="Skip pages already present in the resume JSON (default: off, re-process everything)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        raise SystemExit(1)

    # Derive per-PDF output paths from the input filename stem so that running
    # on different PDFs does not mix up or overwrite each other's results.
    stem = pdf_path.stem
    output_json   = Path(f"{stem}_answers.json")
    output_tex    = Path(f"{stem}_answers.tex")
    output_report = Path(f"{stem}_answers.pdf")
    debug_image_dir = Path(f"debug_crops_{stem}")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env or environment.")
        raise SystemExit(1)

    client = genai.Client(api_key=api_key)

    if SAVE_DEBUG_IMAGES:
        debug_image_dir.mkdir(parents=True, exist_ok=True)

    # Resume support — only used when --skip is passed
    existing = load_existing_results(output_json) if args.skip else {}
    if existing:
        print(f"Resuming -- {len(existing)} pages already done, skipping them.")
    elif args.skip:
        print("No existing results found, processing all pages.")

    print(f"Converting PDF to images at {PDF_DPI} DPI (this may take a minute)...")
    t_pdf_to_images = time.perf_counter()
    pages = convert_from_path(str(pdf_path), dpi=PDF_DPI, thread_count=os.cpu_count())
    pdf_to_images_s = time.perf_counter() - t_pdf_to_images
    n_pages = len(pages)
    per_page = pdf_to_images_s / n_pages if n_pages else 0.0
    print(
        f"PDF→images ({PDF_DPI} DPI): {pdf_to_images_s:.2f}s total "
        f"({per_page:.2f}s/page) — {n_pages} pages.\n"
    )

    results_map: dict[int, dict] = dict(existing)

    for page_num, page in enumerate(pages, start=1):
        if page_num in results_map:
            print(f"  Page {page_num:3d}/{len(pages)} -- skipped (already processed)")
            continue

        print(f"  Page {page_num:3d}/{len(pages)} -- extracting...", end="", flush=True)

        crop = crop_top(page, CROP_TOP_FRACTION)
        img_bytes = to_jpeg_bytes(crop)

        if SAVE_DEBUG_IMAGES:
            crop.save(debug_image_dir / f"page_{page_num:04d}.jpg", quality=85)

        data = call_gemini(client, img_bytes, page_num)
        data["page_number"] = page_num
        results_map[page_num] = data

        conf = data.get("confidence", "?")
        name = data.get("student_name", "?")
        marker = {"high": "OK", "medium": "??", "low": "!!", "failed": "XX"}.get(conf, "??")
        q38lt = data.get("q38_left_top", "?")
        q39l  = data.get("q39_left", "?")
        q40l  = data.get("q40_left", "?")
        q38lb = data.get("q38_left_bottom", "?")
        q39r  = data.get("q39_right", "?")
        q40r  = data.get("q40_right", "?")
        print(f" [{marker}] {name}  |  Q38L↑:{q38lt}  Q39L:{q39l}  Q40L:{q40l}  Q38L↓:{q38lb}  Q39R:{q39r}  Q40R:{q40r}")

        save_results(list(results_map.values()), output_json)
        time.sleep(API_CALL_DELAY_S)

    all_results = list(results_map.values())
    print_summary(all_results)
    generate_report_pdf(all_results, output_tex, output_report)
    print(f"\nJSON  -> {output_json}")
    print(f"LaTeX -> {output_tex}")
    print(f"PDF   -> {output_report}")
    if SAVE_DEBUG_IMAGES:
        print(f"Crops -> {debug_image_dir}/")


if __name__ == "__main__":
    main()
