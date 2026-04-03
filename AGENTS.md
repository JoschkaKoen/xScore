# Auto-Grader — Agent Guide

## Project Overview

Auto-Grader is a Python-based CLI toolchain for processing scanned IGCSE Physics exam papers. It provides two main capabilities:

1. **PDF Preprocessing** (`autograder.py` + `pipeline/scan_deskew.py`): Cleans scanned exam PDFs in three passes: (1) blank/white page removal; (2) 90-degree auto-rotation using Tesseract OSD, with pages copied losslessly via `pikepdf`; (3) optional fine deskew — each A3-portrait page is split into top/bottom A4 halves, sub-degree skew is detected per half via vertical-projection variance (OpenCV), correction is applied at full resolution with bicubic interpolation, and the result is embedded as a rasterised PDF. The `--deskew` CLI flag enables pass 3 from `autograder.py`; `pipeline/pdf_cleanup.py` runs it automatically (pass `deskew=False` to skip).

2. **Answer Extraction** (`scripts/extract_answers.py` + `extraction/` package): Uses Gemini or Kimi vision APIs (see `config.AI_MODEL`) to extract student names and handwritten multiple-choice answers (Questions 38-40), with structured output via Pydantic exam profiles.

The project is designed for educators who need to process batches of scanned exam papers and extract student responses for grading or analysis.

## Technology Stack

- **Language**: Python 3.10 or newer
- **Core Dependencies**:
  - `pdf2image` (≥1.17.0) — PDF to image conversion
  - `pytesseract` (≥0.3.13) — OCR for orientation detection
  - `pikepdf` (≥10.0.0) — Lossless PDF manipulation (passes 1–2)
  - `opencv-python-headless` (≥4.9.0) — Image processing for fine deskew (pass 3)
  - `google-genai` (≥1.0.0) — Gemini Vision API client
  - `openai` (optional) — Kimi / Moonshot (OpenAI-compatible) client
  - `pydantic` (≥2.0.0) — Structured output validation
  - `Pillow` (≥10.0.0) — Image processing
  - `numpy` (≥1.24.0) — Numerical operations for blank detection
  - `python-dotenv` (≥1.0.0) — Environment configuration

- **System Dependencies** (must be installed separately):
  - [Poppler](https://poppler.freedesktop.org/) — `pdftoppm`, etc. (used by `pdf2image`)
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Used for orientation detection
  - XeLaTeX — For generating PDF reports (optional, macOS: `brew install --cask mactex`)

## Project Structure

```
Auto-Grader/
├── autograder.py              # Main PDF cleaning script
├── scripts/                   # Standalone CLIs (not driven by grade.py)
│   ├── extract_answers.py     # Answer extraction CLI
│   ├── bench_pdf_render.py    # Benchmark utility for PDF→image conversion
│   ├── benchmark_eval_loop.py # Repeated eval driver (subprocess CLI)
│   ├── improvement_agent.py   # Iterative tuning helper (optional)
│   ├── visualize_scaffold_boxes.py
│   └── visualize_scan_overlays.py
├── config.py                  # Tunables: AI_MODEL, EXAM_PROFILE, DPI, paths, etc.
├── extraction/                # Answer extraction package
│   ├── profiles/              # ExamProfile: prompt + Pydantic schema + answer_fields
│   ├── providers/             # GeminiProvider, KimiProvider, multi-pass voting
│   ├── images.py              # Crop, preprocess, JPEG, normalize MC letters
│   ├── ground_truth.py        # Load GT, fuzzy names, accuracy
│   ├── reporting.py           # JSON I/O, LaTeX/PDF report, terminal summary
│   └── eval.py                # extract_first_n_students_eval
├── requirements.txt           # Python dependencies
├── README.md                  # User-facing documentation
├── AGENTS.md                  # This file
├── .env                       # API keys and secrets (NOT committed)
├── .gitignore                 # Excludes .env, outputs, debug files
├── Ground Truth               # Tab-separated reference answers (local only)
├── output/                    # Local only — gitignored; never commit or push to GitHub
├── Space Physics Unit Test/         # Source scanned PDFs (gitignored)
├── debug/                     # Debug crops (e.g. debug/debug_crops_*)
└── *_first*_eval.json         # Eval outputs (auto-generated)
```

## Setup and Installation

```bash
# 1. Install system dependencies (macOS example)
brew install poppler tesseract

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env .env.local  # Edit with your keys
# Set GOOGLE_API_KEY or GEMINI_API_KEY (Gemini), or MOONSHOT_API_KEY (Kimi)
```

## Usage

### PDF Preprocessing (autograder.py)

```bash
# Basic usage
python autograder.py input.pdf output.pdf

# With custom parameters
python autograder.py scan.pdf cleaned.pdf \
    --dpi 200 \
    --blank-threshold 248 \
    --blank-std 6
```

**Parameters**:
- `--dpi`: Rasterization DPI for OSD analysis (default: 300)
- `--blank-threshold`: Grayscale mean ≥ this → candidate blank (default: 250)
- `--blank-std`: Grayscale std ≤ this → candidate blank (default: 6)

### Answer Extraction (scripts/extract_answers.py)

```bash
# Process default PDF (``config.DEFAULT_PDF``, e.g. Space Physics scan)
python scripts/extract_answers.py

# Process specific PDF
python scripts/extract_answers.py "Space Physics Unit Test/scan 300dpi.pdf"

# Process first N students only (for evaluation)
python scripts/extract_answers.py --first-students 12

# Resume interrupted run (skip already processed pages)
python scripts/extract_answers.py --skip
```

**Output files** (all under gitignored `output/` — never push; eval JSON in project root):
- `output/extract_answers/<safe_stem>/{stem}_answers.{json,tex,pdf}` — Full extraction runs
- `debug/debug_crops_{stem}/` — Cropped page images (if enabled)

## grade.py prompt parsing

`pipeline/prompt_parser.parse_prompt` calls Kimi with a fixed JSON schema (`TaskInstruction`): `task_type`, `student_filter`, `dpi`, `folder_hint`, `folder_path`, `skip_clean_scan`, `force_clean_scan`, `rescaffold`, `through_step` (1–11 or null), `no_report`. In `grade.py`, CLI flags merge with these: boolean options use **OR**; `--folder`, `--dpi`, and `--through-step` **override** the parsed values when provided. `find_folder` priority: `--folder` → `folder_path` from parse → `folder_hint` → heuristic.

## Code Organization

### autograder.py

Main functions:
- `detect_rotation(image)` → int: Uses Tesseract OSD to detect rotation angle (0°, 90°, 180°, 270°)
- `is_blank_page(image, mean_threshold, std_threshold)` → bool: Detects blank pages using grayscale statistics
- `_osd_worker(page_num, input_path, dpi)` → tuple: Parallel worker for rotation detection
- `process_pdf(input_path, output_path, ...)` → None: Passes 1–2 of the pipeline

Processing pipeline (passes 1–2):
1. Render all pages at low DPI (72) for fast blank detection
2. Filter out pages where mean ≥ threshold AND std ≤ threshold
3. Run parallel OSD at full DPI (300) on content pages only
4. Build output PDF with rotation metadata applied (lossless pikepdf)

CLI flag `--deskew` triggers pass 3 (see `pipeline/scan_deskew.py`) after `process_pdf` completes.

### pipeline/scan_deskew.py

Fine deskew module for A3-portrait scans. **Output PDF is rasterised** at the chosen DPI (no vector text).

Main functions:
- `get_deskew_angle(gray)` → float: Otsu binarise at native resolution, sweep -3°…+3° in 0.01° steps, return best angle by max vertical-projection variance.
- `deskew_image(gray, angle)` → np.ndarray: Apply angle at full resolution with `INTER_CUBIC`; skip if `|angle| < 0.05°`.
- `deskew_page_halves(page_gray)` → (array, top_angle, bot_angle): Split at midpoint, detect and correct each half independently, reassemble.
- `deskew_pdf_raster(input_pdf, output_pdf, dpi, reflines_sidecar=None)` → Path: Render all pages, deskew per page, assemble via PyMuPDF. *input_pdf* and *output_pdf* must resolve to different paths (never overwrite the source). Optional `reflines_sidecar` fixes the JSON path when writing the PDF to a temp file first. Prints per-page angles to console.

### scripts/extract_answers.py + `extraction/`

- **CLI** (`scripts/extract_answers.py`): `main()`, argparse, full-PDF loop; loads dotenv once at startup.
- **Profiles** (`extraction/profiles/`): `ExamProfile` bundles `prompt`, Pydantic `schema`, and `answer_fields`. Selector: `EXAM_PROFILE` in `config.py`.
- **Providers** (`extraction/providers/`): `GeminiProvider`, `KimiProvider`; `get_provider()`, `call_ocr_api()`, `multi_pass_extract()` for eval.
- **Ground truth** (`extraction/ground_truth.py`): Fuzzy name matching and accuracy vs reference file.
- **Eval** (`extraction/eval.py`): `extract_first_n_students_eval()`.

Processing pipeline:
1. Convert PDF to images at `PDF_DPI` (from `config.py`)
2. Crop top fraction (`CROP_TOP_FRACTION`); optional preprocess (contrast/sharpness/brightness)
3. Call configured provider with profile prompt + schema
4. Compare against ground truth (if available)
5. Colored terminal output and LaTeX report

### Configuration Constants

**autograder.py**:
- `ANALYSIS_DPI = 300` — OSD analysis resolution
- `BLANK_DPI = 72` — Fast blank detection resolution
- `BLANK_MEAN_THRESHOLD = 250` — Blank page brightness threshold
- `BLANK_STD_THRESHOLD = 6` — Blank page uniformity threshold

**config.py** (extraction):
- `AI_MODEL` — Gemini or Kimi model id
- `EXAM_PROFILE` — e.g. `igcse_physics`
- `PDF_DPI`, `CROP_TOP_FRACTION`, `PREPROCESS_*`, `GEMINI_*`, `KIMI_*`, `MAX_RETRIES`, `GROUND_TRUTH_PATH`, etc.

## Development Conventions

### Output Naming

`grade.py` writes pipeline artifacts for each exam under `output/<exam_stem>/`: `scaffolds/`, `scaffold_images/`, `overlays/` (vector scaffold-box PDFs), `trials/` (optional ad-hoc deskew experiments), `cleaned_scan.pdf` plus deskew sidecar and scan debug PDFs at the artifact root, and `runs/<timestamp>/` for LaTeX/PDF reports.

Full-run extraction (`scripts/extract_answers.py`) writes under `output/extract_answers/<safe_pdf_stem>/`:
- Input example: `Space Physics Unit Test/scan 400dpi.pdf`
- JSON / TeX / PDF: `output/extract_answers/scan_400dpi/scan 400dpi_answers.{json,tex,pdf}` (original stem preserved in filenames)
- Debug crops: `debug/debug_crops_{stem}/`
- Eval (`--first-students N`): `{stem}_firstN_eval.json` in the project root

This ensures processing different PDFs doesn't overwrite previous results.

### Debug Image Handling

Set `SAVE_DEBUG_IMAGES = True` in `config.py` to save cropped page images. Debug directories are gitignored.

### Ground Truth Format

The `Ground Truth` file (trailing space in filename) is a tab-separated file:

```
Ground Truth 
Yuze     A D B C A C
Simon    A A B C B C
...
```

Format: `Name  Q38_LT  Q39_L  Q40_L  Q38_LB  Q39_R  Q40_R`

### Confidence Levels

Extraction uses three confidence tiers:
- **high**: Clear, unambiguous markings
- **medium**: Somewhat faint but readable, minor ambiguity
- **low**: Significant ambiguity, very faint, or multiple competing marks

## Testing and Evaluation

### Benchmarking PDF Rendering

```bash
python scripts/bench_pdf_render.py [path/to.pdf] --dpi 300
```

Times the PDF→image conversion without API calls.

### Evaluation Mode

```bash
python scripts/extract_answers.py --first-students 12
```

Processes only first N pages and compares against ground truth, printing:
- Per-student accuracy with color-coded terminal output
- Cumulative accuracy across all processed students
- Per-field confidence indicators

### Accuracy Calculation

Accuracy is calculated against ground truth using fuzzy name matching (SequenceMatcher ≥ 60% similarity). Each of the 6 answer fields contributes equally to the score.

## Security Considerations

### API Key Management

- API keys are stored in `.env` file (gitignored)
- Supported variables: `GOOGLE_API_KEY`, `GEMINI_API_KEY`
- `GOOGLE_API_KEY` takes precedence if both are set
- Never commit `.env` or any file containing secrets

### Data Privacy

- Scanned exam PDFs contain student names and work — do not commit to public repositories
- The `Space Physics Unit Test/` directory is gitignored
- Ground truth files are gitignored (`Ground Truth*`)
- Debug crop images may contain student information

### LaTeX Compilation

The `generate_report_pdf()` function calls `xelatex` with `-interaction=nonstopmode`. Ensure input data is sanitized to prevent LaTeX injection (the `_tex_escape()` function handles common escape sequences).

## Git Workflow

```bash
# Pre-commit checklist
git status
# Ensure no .env files, no PDF scans, no debug crops are staged

git add autograder.py scripts/extract_answers.py  # etc.
git commit -m "Your message"
```

**Never commit**:
- `.env` or `.env.*`
- Any `*.pdf` (gitignored repo-wide; scans and generated reports)
- `Space Physics Unit Test/`
- `output/` directory (entire tree — do not `git add -f output/`; nothing under it belongs on GitHub)
- `debug_crops*/` directories
- `*_answers.{json,tex,pdf}`
- `Ground Truth*` files

## Troubleshooting

### Tesseract OSD failures

If rotation detection fails consistently, check:
- Tesseract is installed: `tesseract --version`
- Image quality is sufficient (try higher `--dpi`)

### Gemini API errors

- Verify `GOOGLE_API_KEY` is set in `.env`
- Check API quota and rate limits
- Enable retry with exponential backoff is built-in (3 retries default)

### Poppler errors (pdf2image)

- macOS: `brew install poppler`
- Ubuntu/Debian: `sudo apt install poppler-utils`
- Windows: Add poppler `bin/` to PATH

### LaTeX compilation fails

- Install XeLaTeX: `brew install --cask mactex` (macOS)
- Ensure Chinese font support (uses PingFang SC)
- Check `xelatex` is in PATH
