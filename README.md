# Auto-Grader

Python CLI tools for **cleaning scanned exam PDFs**, **extracting handwritten answers** with vision models (Gemini or Kimi), and **grading arbitrary exams** from a natural-language prompt.

## Features

| Tool | Purpose |
|------|---------|
| **`autograder.py`** | Auto-rotate pages (Tesseract OSD), remove blank pages, write a cleaned PDF (`pikepdf`). |
| **`extract_answers.py`** | Benchmark path: fixed IGCSE Physics profile (Q38–Q40 MC), JSON/LaTeX/PDF reports, optional ground-truth eval. |
| **`grade.py`** | Generic pipeline: NL prompt → exam folder → roster (Excel) → AI scaffold → page assignment → grading → terminal + PDF report. |

## Requirements

- **Python** 3.10+
- **System** (not from pip):
  - [Poppler](https://poppler.freedesktop.org/) — `pdf2image`
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — orientation in `autograder.py`
  - **XeLaTeX** (optional) — PDF reports (`xelatex` on `PATH`; e.g. macOS: MacTeX / `brew install --cask mactex-no-gui`)

### Installing system dependencies

**macOS (Homebrew):**

```bash
brew install poppler tesseract
```

**Debian / Ubuntu:**

```bash
sudo apt install poppler-utils tesseract-ocr
```

## Setup

```bash
cd /path/to/Auto-Grader
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (not committed) and set API keys:

- **Gemini:** `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- **Kimi (Moonshot):** `KIMI_API_KEY`

See `config.py` for model selection (`AI_MODEL`), DPI, crop, and pipeline defaults.

## Usage

### 1. Clean a scan (`autograder.py`)

```bash
python autograder.py input.pdf output.pdf
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dpi` | `300` | Rasterization DPI for OSD (higher = sharper, slower) |
| `--blank-threshold` | `250` | Grayscale mean ≥ this → candidate blank (0–255) |
| `--blank-std` | `6` | Grayscale std ≤ this → candidate blank |

```bash
python autograder.py scan.pdf cleaned.pdf --dpi 200 --blank-threshold 248 --blank-std 6
```

### 2. Extract answers — benchmark / IGCSE (`extract_answers.py`)

Uses the configured **`EXAM_PROFILE`** and vision model from `config.py`. Outputs under `output/` (full run) or eval JSON in the project root (`--first-students`).

```bash
python extract_answers.py                              # default PDF from config
python extract_answers.py "path/to/scan.pdf"
python extract_answers.py --first-students 12            # eval first N pages vs ground truth
python extract_answers.py --skip                         # resume from existing JSON
python extract_answers.py --version
```

Ground truth: tab-separated file at `GROUND_TRUTH_PATH` in `config.py` (repo root by default).

### 3. Generic grading (`grade.py`)

End-to-end grading from a **natural language prompt**. Expects an exam folder with e.g. raw exam PDF, answer key, scan PDFs, and `StudentList.xlsx` (see `pipeline/` and `AGENTS.md`).

```bash
python grade.py "check all multiple choice question answers"
python grade.py "count marks for each student" --folder "Space Physics Unit Test"
python grade.py "check the first 5 students' answers" --dpi 300
```

| Option | Description |
|--------|-------------|
| `--folder PATH` | Exam folder (overrides auto-discovery) |
| `--dpi N` | Override rendering DPI |
| `--no-cleanup` | Skip `autograder`-style cleaning; use existing `cleaned_scan.pdf` or a scan |
| `--output-dir DIR` | Where to write `{folder}_grade_report.{tex,pdf}` (default: `output/`) |
| `--no-report` | Terminal only; skip LaTeX/PDF |

**Ground truth in the exam folder:** if a file such as `ground_truth.txt` or `Ground Truth ` is found, results are compared and a **colour-coded accuracy summary** is printed in the terminal; the same evaluation is included in the PDF report when LaTeX succeeds.

## Project layout (high level)

```
autograder.py          # PDF cleaning
extract_answers.py     # CLI for IGCSE-style extraction benchmark
grade.py               # Generic grading pipeline CLI
config.py              # Tunables (models, DPI, paths, pipeline defaults)
extraction/            # Profiles, providers (Gemini/Kimi), eval, reporting
pipeline/              # grade.py: folder discovery, scaffold, grading, GT, PDF report
version.py             # Release version string
AGENTS.md              # Detailed structure for contributors / AI agents
```

## Security & privacy

- Do not commit `.env`, scans, or ground-truth files with student data.  
- `*.pdf` and exam folders are gitignored by policy; keep scans local.

## License / docs

For deeper conventions (output naming, eval JSON, LaTeX), see **`AGENTS.md`**.
