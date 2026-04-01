# Auto-Grader

Python CLI tools for **cleaning scanned exam PDFs**, **extracting handwritten answers** with vision models (Gemini or Kimi), and **grading arbitrary exams** from a natural-language prompt.

## Overview — what this program is

Auto-Grader is aimed at **educators and markers** who work with **stacks of scanned exam papers**. Scans are often mis-rotated, contain blank separator pages, and carry handwritten responses that are tedious to transcribe or check by hand. This repository bundles three complementary workflows:

1. **Deterministic PDF hygiene** — No AI: fix orientation and drop empty pages so downstream steps see consistent, upright pages.
2. **Profiled extraction** — A fixed **exam profile** (currently oriented around IGCSE Physics-style layouts) tells the model exactly which fields to read (e.g. name, specific multiple-choice cells). Results are validated with **Pydantic** schemas and can be compared to a **ground truth** file for benchmarking.
3. **Prompt-driven grading** — For **any** exam layout you place in a folder (question paper, answer key, class scan, roster), a **natural-language instruction** drives a multi-stage pipeline: the model infers task type, student subset, and DPI; builds a **scaffold** of questions from the PDFs; maps scan pages to students; detects which questions were attempted; grades against the key; optionally compares to a folder-local ground truth; and emits a **terminal summary** plus optional **XeLaTeX PDF report**.

The **vision layer** (Gemini or Kimi, selectable in `config.py`) receives **cropped or full-page images** derived from PDFs. **`grade.py`** currently wires the generic pipeline to **Kimi** (`KIMI_API_KEY`). **`extract_answers.py`** uses **`create_extraction_client()`** and respects **`AI_MODEL`** for Gemini vs Kimi.

**Important distinction:** Ground truth for **`extract_answers.py`** is configured via **`GROUND_TRUTH_PATH`** in `config.py` (extraction’s own loader and fuzzy name matching). Ground truth for **`grade.py`** lives **inside the exam folder** and is handled by **`pipeline/ground_truth.py`** against pipeline **`StudentResult`** data — the two evaluators are separate.

## Components at a glance

| Tool | Purpose |
|------|---------|
| **`autograder.py`** | Auto-rotate pages (Tesseract OSD), remove blank pages, write a cleaned PDF (`pikepdf`). |
| **`extract_answers.py`** | Benchmark path: fixed IGCSE Physics profile (Q38–Q40 MC), JSON/LaTeX/PDF reports, optional ground-truth eval. |
| **`grade.py`** | Generic pipeline: NL prompt → exam folder → roster (Excel) → AI scaffold → page assignment → grading → terminal + PDF report. |

---

## How each tool works (process)

### A. PDF cleaning — `autograder.py`

This path is **fully local** (Poppler + Tesseract + NumPy + PikePDF). It does **not** call cloud APIs.

1. **Blank detection (fast pass)** — Every page is rendered at **low DPI** (72). Each page is converted to grayscale; if the mean brightness is high enough **and** the standard deviation is low enough, the page is treated as **blank** and dropped from the pipeline.
2. **Orientation (content pages only)** — Remaining pages are analyzed at **configurable DPI** (default 300). **Tesseract OSD** estimates rotation; the script rotates so text reads upright (with a confidence guard so uncertain pages stay unchanged).
3. **Output** — Selected pages are assembled into a **new PDF** with **`pikepdf`**, copying page objects **without re-encoding** the underlying streams where possible, so quality is preserved compared to a full raster re-export.

**Typical use:** Run this on a raw batch scan **before** extraction or before placing the scan in an exam folder for `grade.py` (the pipeline can also run the same style of cleanup for you unless you pass `--no-cleanup`).

---

### B. IGCSE-style extraction — `extract_answers.py`

This path is **layout-specific**: **`EXAM_PROFILE`** in `config.py` selects a **prompt + JSON schema + answer field list** under `extraction/profiles/`. The default profile targets **handwritten multiple-choice** for defined question regions (e.g. Questions 38–40).

1. **Input** — A single scan PDF (default path from `DEFAULT_PDF` in `config`, or a path you pass on the command line).
2. **Rasterize** — Pages are converted to images at **`PDF_DPI`** via `pdf2image`.
3. **Crop and preprocess** — The top portion of each page is kept (**`CROP_TOP_FRACTION`**); optional contrast/sharpness/brightness steps prepare pixels for the model.
4. **Vision call** — For each page, JPEG bytes are sent to **Gemini or Kimi** with the profile prompt. The API returns JSON matching the profile’s **Pydantic** model; answers are **normalized** (e.g. MC letters).
5. **Outputs**
   - **Full run:** JSON + LaTeX + compiled PDF under `output/` (stem derived from the input PDF name). Optional **debug crops** if enabled in config.
   - **Eval shortcut (`--first-students N`):** Only the first **N** pages are processed; results are compared to the ground-truth file at **`GROUND_TRUTH_PATH`**; coloured terminal accuracy and a small eval JSON in the project root.
6. **Resume (`--skip`)** — Merges with an existing results JSON so you can continue after interruptions without redoing finished pages.

---

### C. Generic grading — `grade.py`

You describe **what** you want in English (e.g. “check all multiple choice”, “only the first 5 students”). The program **parses** that into structured instructions, then runs a fixed sequence of stages that share one **Kimi** client.

| Step | What happens |
|------|----------------|
| **1** | **Parse prompt** — `pipeline/prompt_parser.py` calls the model to produce a **`TaskInstruction`**: task type, which students to include, optional folder hint, and DPI. |
| **2** | **Resolve exam folder** — `find_folder()` uses the hint and heuristics, or **`--folder`** overrides everything. |
| **3** | **Roster** — `StudentList.xlsx` in that folder is read (`openpyxl`); student names define who should appear in the report. |
| **4** | **Scaffold** — PDFs in the folder (exam paper, answer key, etc.) are analyzed to build an **`ExamScaffold`**: numbered questions, types, and marking logic the grader will use. |
| **5** | **Scan preparation** — Unless **`--no-cleanup`**, the class scan PDF is cleaned (same idea as `autograder.py`) and written as **`cleaned_scan.pdf`** in the folder. With **`--no-cleanup`**, an existing cleaned scan or a filename containing `scan` is used. |
| **6** | **Page assignment** — For each scan page, the **name region** is cropped and classified so pages are **linked to roster students** (vision + roster order). |
| **7** | **Exercise detection** — Per student page, the model decides **which scaffold questions** were actually attempted or visible. |
| **8** | **Grading** — For each student and relevant question, page crops and the scaffold drive **comparison to the answer key**; **`StudentResult`** rows accumulate scores and rationale. |
| **9** | **Terminal output** — Tables and summaries print to the console (scaffold, pages, exercises, marks, grand total). |
| **10** | **Ground truth (optional)** — If a recognised file exists in the folder (e.g. `ground_truth.txt`, or `Ground Truth ` with trailing space), results are compared and a **colour-coded** accuracy block prints. If the file is missing or unparsable, this step is skipped with a message. |
| **11** | **PDF report (optional)** — Unless **`--no-report`**, LaTeX is written to **`{output_dir}/{folder_stem}_grade_report.tex`** and **`xelatex`** compiles **`_grade_report.pdf`**. Evaluation stats are embedded when step 10 succeeded. |

Configuration that especially affects this pipeline includes **`PIPELINE_DEFAULT_DPI`**, **`NAME_RECOGNITION_DPI`**, **`NAME_CROP_FRACTION`**, and related keys in **`config.py`**.

---

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
| `--output-dir DIR` | Where to write `{folder_stem}_grade_report.{tex,pdf}` (default: `output/`) |
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

## Further reading

For dependency versions, file naming conventions, eval JSON layout, and LaTeX details, see **`AGENTS.md`**.
