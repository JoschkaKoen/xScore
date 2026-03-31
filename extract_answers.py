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
from difflib import SequenceMatcher

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
# Ground Truth Handling
# ---------------------------------------------------------------------------

# Default path to ground truth file (with trailing space as created)
GROUND_TRUTH_PATH = Path("/Users/joschka/Desktop/Programming/Auto-Grader/Ground Truth ")

# Answer field names in order (matching ground truth columns)
ANSWER_FIELDS = [
    "q38_left_top",
    "q39_left",
    "q40_left",
    "q38_left_bottom",
    "q39_right",
    "q40_right",
]


def load_ground_truth(gt_path: Path = GROUND_TRUTH_PATH) -> dict[str, list[str]]:
    """Load ground truth answers from file.
    
    Returns dict mapping student_name -> [q38_left_top, q39_left, q40_left, 
                                          q38_left_bottom, q39_right, q40_right]
    Name matching is fuzzy to handle slight variations.
    """
    if not gt_path.exists():
        return {}
    
    gt_data = {}
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header line
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 7:  # name + 6 answers
                name = parts[0]
                answers = parts[1:7]  # Take exactly 6 answers
                gt_data[name] = answers
    except Exception as e:
        print(f"Warning: Could not load ground truth from {gt_path}: {e}")
    
    return gt_data


def fuzzy_match_name(extracted_name: str, gt_names: list[str]) -> str | None:
    """Find the best matching ground truth name using fuzzy matching."""
    if not extracted_name or extracted_name in ("UNKNOWN", "EXTRACTION_ERROR"):
        return None
    
    extracted_lower = extracted_name.lower().strip()
    
    # First try exact match (case-insensitive)
    for gt_name in gt_names:
        if gt_name.lower() == extracted_lower:
            return gt_name
    
    # Try substring matching
    for gt_name in gt_names:
        gt_lower = gt_name.lower()
        if extracted_lower in gt_lower or gt_lower in extracted_lower:
            return gt_name
    
    # Try fuzzy matching with similarity ratio
    best_match = None
    best_ratio = 0.0
    for gt_name in gt_names:
        ratio = SequenceMatcher(None, extracted_lower, gt_name.lower()).ratio()
        if ratio > best_ratio and ratio >= 0.6:  # 60% similarity threshold
            best_ratio = ratio
            best_match = gt_name
    
    return best_match


def calculate_student_accuracy(extracted: dict, gt_answers: list[str]) -> float:
    """Calculate accuracy percentage for a single student."""
    correct = 0
    total = len(ANSWER_FIELDS)
    
    for i, field in enumerate(ANSWER_FIELDS):
        extracted_val = extracted.get(field, "?").upper().strip()
        gt_val = gt_answers[i].upper().strip() if i < len(gt_answers) else ""
        
        if extracted_val == gt_val and extracted_val not in ("", "?"):
            correct += 1
    
    return (correct / total) * 100 if total > 0 else 0.0


def format_accuracy(acc: float) -> str:
    """Format accuracy as percentage string with color coding via ASCII."""
    if acc >= 80:
        return f"{acc:.0f}%"  # Good
    elif acc >= 50:
        return f"{acc:.0f}%"  # Okay
    else:
        return f"{acc:.0f}%"  # Poor

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_PDF = "output/20260330135527722.pdf"
SAVE_DEBUG_IMAGES = True

PDF_DPI = 300  # 150 DPI is sufficient for Gemini to read handwritten letters and names
JPEG_QUALITY = 95
CROP_TOP_FRACTION = 0.5

# Gemini 3 Pro Preview is deprecated (shut down); use 3.1 Pro Preview per
# https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL = "gemini-3.1-pro-preview"
API_CALL_DELAY_S = 0
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1


# ---------------------------------------------------------------------------
# Structured output schema — enforced at the token level by Gemini
# ---------------------------------------------------------------------------

class StudentAnswers(BaseModel):
    student_name: str
    student_name_confidence: str  # high, medium, low for the name
    q38_left_top: str             # left side, position 1 (top)     — Q38
    q38_left_top_confidence: str  # high, medium, low
    q39_left: str                 # left side, position 2           — Q39
    q39_left_confidence: str      # high, medium, low
    q40_left: str                 # left side, position 3           — Q40
    q40_left_confidence: str      # high, medium, low
    q38_left_bottom: str          # left side, position 4 (bottom)  — Q38 again
    q38_left_bottom_confidence: str  # high, medium, low
    q39_right: str                # right side, position 1 (top)    — Q39
    q39_right_confidence: str     # high, medium, low
    q40_right: str                # right side, position 2          — Q40
    q40_right_confidence: str     # high, medium, low
    confidence: str               # overall page confidence (high/medium/low)


PROMPT = """\
You are an expert exam grader analyzing scanned IGCSE Physics answer sheets. Your task is to accurately extract student names and multiple-choice answers from handwritten exam papers.

=== PAGE LAYOUT ===
The answer sheet is divided into sections:

TOP SECTION (approximately top 15% of page):
  - Student name written by hand in English letters
  - Usually appears near the top-left or top-center
  - May be preceded by labels like "Name:", "Student:", or similar

LEFT COLUMN (middle-left area, vertical arrangement):
  Position 1 (upper): Question 38  → field: q38_left_top
  Position 2:         Question 39  → field: q39_left
  Position 3:         Question 40  → field: q40_left
  Position 4 (lower): Question 38  → field: q38_left_bottom (second instance of Q38)

RIGHT COLUMN (middle-right area, vertical arrangement):
  Position 1 (upper): Question 39  → field: q39_right
  Position 2 (lower): Question 40  → field: q40_right

=== ANSWER FORMAT GUIDE ===
Students indicate answers in these ways:
1. CIRCLING the letter (A, B, C, or D) on a printed grid
2. WRITING the letter clearly next to the question number
3. TICKING or marking the chosen option

Look for:
- Printed letters A B C D arranged horizontally or vertically
- One option will have a circle, tick, cross, or handwritten mark
- The mark may be: a circle (O), tick (✓), cross (X), underline, or scribble over the letter

=== HANDWRITING RECOGNITION TIPS ===
For student names:
- Common patterns: First Last, First M. Last, Last First
- Look for capitalized words
- Ignore titles like "Mr.", "Ms.", "Miss" if present
- If multiple names present, choose the one that appears to be the student's full name

=== HANDLING AMBIGUOUS CASES ===
1. Multiple answers marked:
   - If student circled/changed answer: pick the FINAL/clearest answer
   - If two answers equally prominent: return "?" with low confidence

2. Crossed-out answers:
   - Ignore crossed-out responses, use the replacement answer
   - If unclear which is final: return "?" with low confidence

3. Stray marks:
   - Distinguish between intentional answers and accidental marks
   - A clear circle/tick near a letter = intentional answer
   - Random dots/lines far from options = ignore

4. Poor image quality:
   - Look for contrast differences (darker areas = ink)
   - If answer is faint but discernible: use it with medium confidence
   - If completely unreadable: return "?" with low confidence

5. Partial marks:
   - Half-circle around letter = that letter was selected
   - Letter written small nearby = that letter was selected

=== EXTRACTION RULES ===
1. For each question field, return ONLY: A, B, C, D, or ?
   - NEVER return descriptive text like "circle around B" - just "B"
   - NEVER return empty string "" - use "?" if unreadable

2. For student_name field:
   - Return the name EXACTLY as written (preserve spelling)
   - Convert to proper case if all caps or all lowercase
   - Return "UNKNOWN" only if completely illegible or missing
   - Do NOT include labels like "Name:" - just the name itself

3. Confidence assessment (per-field):
   HIGH confidence when:
   - Letter is clearly written or circled with dark, unambiguous ink
   - No competing marks nearby
   - Readable even at a glance
   
   MEDIUM confidence when:
   - Letter is somewhat faint but readable
   - Minor ambiguity (could be B or D, but B is more likely)
   - Slightly messy handwriting but decipherable
   
   LOW confidence when:
   - Significant ambiguity between two letters
   - Very faint or smudged marking
   - Competing marks that create doubt
   - Any uncertainty that affects grading accuracy

4. Overall page confidence:
   - HIGH: All fields clear and unambiguous
   - MEDIUM: Most fields clear, 1-2 minor uncertainties
   - LOW: Multiple uncertainties or poor image quality

=== EXAMPLES ===
Example 1 - Clear answer:
  [Image shows dark circle around letter B]
  → q39_left: "B", q39_left_confidence: "high"

Example 2 - Ambiguous mark:
  [Image shows faint tick mark between C and D]
  → q40_right: "?", q40_right_confidence: "low"

Example 3 - Changed answer:
  [Image shows crossed-out A, circle around C]
  → q38_left_top: "C", q38_left_top_confidence: "medium"

Example 4 - Written answer:
  [Image shows handwritten "D" next to question number]
  → q39_left: "D", q39_left_confidence: "high"

Example 5 - Name extraction:
  [Image shows "john smith" written at top]
  → student_name: "John Smith", student_name_confidence: "high"
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
        "student_name_confidence": "failed",
        "q38_left_top": "",
        "q38_left_top_confidence": "failed",
        "q39_left": "",
        "q39_left_confidence": "failed",
        "q40_left": "",
        "q40_left_confidence": "failed",
        "q38_left_bottom": "",
        "q38_left_bottom_confidence": "failed",
        "q39_right": "",
        "q39_right_confidence": "failed",
        "q40_right": "",
        "q40_right_confidence": "failed",
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


def print_summary(results: list[dict], ground_truth: dict[str, list[str]] | None = None):
    """Print a quick summary of extraction quality and accuracy."""
    total = len(results)
    high = sum(1 for r in results if r.get("confidence") == "high")
    medium = sum(1 for r in results if r.get("confidence") == "medium")
    low = sum(1 for r in results if r.get("confidence") == "low")
    failed = sum(1 for r in results if r.get("confidence") == "failed")
    unknown = sum(1 for r in results if r.get("student_name") in ("UNKNOWN", "EXTRACTION_ERROR"))
    
    # Calculate overall accuracy if ground truth available
    if ground_truth:
        gt_names = list(ground_truth.keys())
        total_correct = 0
        total_answer_fields = 0
        matched_students = 0
        
        for r in results:
            name = r.get("student_name", "")
            if name not in ("UNKNOWN", "EXTRACTION_ERROR", ""):
                matched_gt_name = fuzzy_match_name(name, gt_names)
                if matched_gt_name:
                    matched_students += 1
                    gt_answers = ground_truth[matched_gt_name]
                    for i, field in enumerate(ANSWER_FIELDS):
                        extracted_val = r.get(field, "?").upper().strip()
                        gt_val = gt_answers[i].upper().strip() if i < len(gt_answers) else ""
                        total_answer_fields += 1
                        if extracted_val == gt_val and extracted_val not in ("", "?"):
                            total_correct += 1
        
        overall_acc = (total_correct / total_answer_fields * 100) if total_answer_fields > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION SUMMARY: {total} pages processed")
    print(f"  High confidence:   {high}")
    print(f"  Medium confidence: {medium}")
    print(f"  Low confidence:    {low}")
    print(f"  Failed:            {failed}")
    print(f"  Unreadable names:  {unknown}")
    
    if ground_truth:
        print(f"\n  ACCURACY SUMMARY:")
        print(f"  Students matched to ground truth: {matched_students}/{len(gt_names)}")
        print(f"  Overall accuracy: {overall_acc:.1f}% ({total_correct}/{total_answer_fields} correct)")
    
    print(f"{'=' * 60}")


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
    
    # Load ground truth for accuracy calculation
    ground_truth = load_ground_truth()
    gt_names = list(ground_truth.keys())
    if ground_truth:
        print(f"Ground truth loaded: {len(ground_truth)} students ({', '.join(gt_names)})")
    else:
        print("Warning: No ground truth found. Accuracy metrics disabled.")

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
    
    # Track cumulative accuracy across all students
    cumulative_correct = 0
    cumulative_total = 0
    students_with_gt = 0

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
        # Per-field confidence indicators
        nc = data.get("student_name_confidence", "?")[0].upper() if data.get("student_name_confidence") else "?"
        c38lt = data.get("q38_left_top_confidence", "?")[0].upper() if data.get("q38_left_top_confidence") else "?"
        c39l = data.get("q39_left_confidence", "?")[0].upper() if data.get("q39_left_confidence") else "?"
        c40l = data.get("q40_left_confidence", "?")[0].upper() if data.get("q40_left_confidence") else "?"
        c38lb = data.get("q38_left_bottom_confidence", "?")[0].upper() if data.get("q38_left_bottom_confidence") else "?"
        c39r = data.get("q39_right_confidence", "?")[0].upper() if data.get("q39_right_confidence") else "?"
        c40r = data.get("q40_right_confidence", "?")[0].upper() if data.get("q40_right_confidence") else "?"
        
        # Calculate accuracy against ground truth
        student_acc_str = "N/A"
        cumulative_acc_str = "N/A"
        
        if ground_truth and name not in ("UNKNOWN", "EXTRACTION_ERROR", "?"):
            matched_gt_name = fuzzy_match_name(name, gt_names)
            if matched_gt_name:
                gt_answers = ground_truth[matched_gt_name]
                student_acc = calculate_student_accuracy(data, gt_answers)
                student_acc_str = format_accuracy(student_acc)
                
                # Update cumulative stats
                for i, field in enumerate(ANSWER_FIELDS):
                    extracted_val = data.get(field, "?").upper().strip()
                    gt_val = gt_answers[i].upper().strip() if i < len(gt_answers) else ""
                    cumulative_total += 1
                    if extracted_val == gt_val and extracted_val not in ("", "?"):
                        cumulative_correct += 1
                
                students_with_gt += 1
                cumulative_acc = (cumulative_correct / cumulative_total * 100) if cumulative_total > 0 else 0
                cumulative_acc_str = format_accuracy(cumulative_acc)
                
                # Store accuracy in data for JSON output
                data["student_accuracy"] = student_acc
                data["matched_ground_truth_name"] = matched_gt_name
        
        print(f" [{marker}] {name}({nc})  |  Q38L↑:{q38lt}({c38lt})  Q39L:{q39l}({c39l})  Q40L:{q40l}({c40l})  Q38L↓:{q38lb}({c38lb})  Q39R:{q39r}({c39r})  Q40R:{q40r}({c40r})  |  Acc: {student_acc_str} (Student) / {cumulative_acc_str} (Cumulative)")

        save_results(list(results_map.values()), output_json)
        time.sleep(API_CALL_DELAY_S)

    all_results = list(results_map.values())
    print_summary(all_results, ground_truth if ground_truth else None)
    generate_report_pdf(all_results, output_tex, output_report)
    print(f"\nJSON  -> {output_json}")
    print(f"LaTeX -> {output_tex}")
    print(f"PDF   -> {output_report}")
    if SAVE_DEBUG_IMAGES:
        print(f"Crops -> {debug_image_dir}/")


if __name__ == "__main__":
    main()
