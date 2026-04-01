# Auto-Grader

Clean scanned exam PDFs, read handwritten answers with **Gemini** or **Kimi**, or run **full exam grading** from a short English prompt.

---

## Contents

- [Which tool do I need?](#which-tool-do-i-need)
- [What the project does](#what-the-project-does)
- [How each workflow runs](#how-each-workflow-runs)
- [Requirements & setup](#requirements)
- [Command-line usage](#usage)
- [Ground truth (two places)](#ground-truth-two-places)
- [Project layout](#project-layout-high-level)

---

## Which tool do I need?

| You want to… | Use |
|--------------|-----|
| Fix rotation and remove blank pages only (no AI, no API) | **`autograder.py`** |
| Extract fixed layout fields (e.g. IGCSE Physics MC) and get JSON / reports / benchmark vs a reference file | **`extract_answers.py`** |
| Grade a whole exam folder from a prompt (roster, scaffold, marks, optional PDF report) | **`grade.py`** |

---

## What the project does

Auto-Grader helps **teachers and markers** who work with **scanned exams**. Typical problems: pages scanned sideways, blank separator sheets, and handwriting that is slow to transcribe or check by eye.

The repo offers **three** paths:

1. **PDF hygiene** — Rotate pages and drop blanks using **Tesseract** and **`pikepdf`**. No cloud calls.
2. **Profiled extraction** — An **exam profile** (see `extraction/profiles/`) defines what the model must read (names, specific boxes, etc.). Output is checked with **Pydantic**. You can score accuracy against a **ground truth** file.
3. **Prompt-driven grading** — Put exam PDFs, answer key, class scan, and **`StudentList.xlsx`** in one folder. You type what you want (e.g. “check all multiple choice”). The app parses that, builds a question list from the PDFs, matches pages to students, grades, and can print a **PDF report** (XeLaTeX).

**APIs:** Vision calls use images from your PDFs. Tune models and DPI in **`config.py`**.

- **`grade.py`** uses **Kimi** only (needs **`KIMI_API_KEY`**).
- **`extract_answers.py`** follows **`AI_MODEL`** and can use **Gemini** or **Kimi**.

---

## How each workflow runs

### `autograder.py` — clean a PDF

Runs entirely on your machine (Poppler, Tesseract, NumPy, PikePDF).

1. Render every page at **low DPI** and mark pages that look uniformly bright as **blank**; those are skipped.
2. For remaining pages, run **Tesseract OSD** at higher DPI (default **300**) to detect rotation; rotate to upright when confidence is good enough.
3. Write a **new PDF** with **`pikepdf`**, reusing page streams where possible so quality stays high.

You can run this **before** `extract_answers.py` or before using a folder with **`grade.py`**. The grading pipeline can also clean for you unless you pass **`--no-cleanup`**.

### `extract_answers.py` — benchmark / IGCSE-style extraction

**Layout is fixed** by **`EXAM_PROFILE`** in `config.py` (prompt + schema + field list under `extraction/profiles/`). The default profile targets handwritten multiple choice in defined regions (e.g. Q38–Q40).

1. Load one scan PDF (`DEFAULT_PDF` or a path you pass).
2. Turn pages into images at **`PDF_DPI`**.
3. **Crop** the top of each page (`CROP_TOP_FRACTION`); optional **preprocess** (contrast, etc.).
4. Send JPEGs to **Gemini or Kimi**; merge responses into the profile’s **Pydantic** model and **normalize** answers (e.g. MC letters).
5. **Full run:** write JSON + LaTeX + PDF under **`output/`**. **`--first-students N`:** only first *N* pages, compare to ground truth, print coloured accuracy. **`--skip`:** resume from existing JSON.

### `grade.py` — generic grading from a prompt

You write plain English; the app turns it into structured settings, then runs these stages (one shared **Kimi** client):

1. **Parse prompt** — Task type, student filter, folder hint, DPI (`pipeline/prompt_parser.py`).
2. **Exam folder** — Resolve path from hint or use **`--folder`**.
3. **Roster** — Read **`StudentList.xlsx`** in that folder.
4. **Scaffold** — Infer questions and marking from PDFs → **`ExamScaffold`**.
5. **Scan PDF** — Clean scan → **`cleaned_scan.pdf`**, or with **`--no-cleanup`** use an existing cleaned/scan file.
6. **Page → student** — Match each page to a roster name (name region + vision).
7. **Which questions** — Per page, detect which scaffold questions were attempted.
8. **Grade** — Compare to the key; build **`StudentResult`** rows.
9. **Print** — Tables and totals in the terminal.
10. **Ground truth** — If a GT file exists in the **folder**, print accuracy (optional; see [below](#ground-truth-two-places)).
11. **Report** — Unless **`--no-report`**, emit **`{folder_stem}_grade_report.{tex,pdf}`** under **`--output-dir`** (default `output/`).

For DPI and name cropping, see **`PIPELINE_DEFAULT_DPI`**, **`NAME_RECOGNITION_DPI`**, **`NAME_CROP_FRACTION`** in `config.py`.

---

## Requirements

- **Python** 3.10+
- **System** (install separately):
  - [Poppler](https://poppler.freedesktop.org/) — for `pdf2image`
  - [Tesseract](https://github.com/tesseract-ocr/tesseract) — for `autograder.py` rotation
  - **XeLaTeX** (optional) — for compiled PDF reports; e.g. macOS: `brew install --cask mactex-no-gui`

**macOS:** `brew install poppler tesseract`  

**Debian / Ubuntu:** `sudo apt install poppler-utils tesseract-ocr`

## Setup

```bash
cd /path/to/Auto-Grader
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create **`.env`** in the project root (never commit it):

| Provider | Variable |
|----------|----------|
| Google Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Kimi (Moonshot) | `KIMI_API_KEY` |

`config.py` holds **`AI_MODEL`**, DPI, crops, and pipeline defaults.

---

## Usage

### 1. `autograder.py`

```bash
python autograder.py input.pdf output.pdf
python autograder.py scan.pdf cleaned.pdf --dpi 200 --blank-threshold 248 --blank-std 6
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--dpi` | 300 | DPI for orientation detection |
| `--blank-threshold` | 250 | Mean grayscale ≥ this → blank candidate |
| `--blank-std` | 6 | Std dev ≤ this → blank candidate |

### 2. `extract_answers.py`

Uses **`EXAM_PROFILE`** and **`AI_MODEL`** from `config.py`.

```bash
python extract_answers.py
python extract_answers.py "path/to/scan.pdf"
python extract_answers.py --first-students 12
python extract_answers.py --skip
python extract_answers.py --version
```

Full runs write under **`output/`**; eval mode also writes a small JSON in the repo root.

### 3. `grade.py`

Needs an exam folder (papers, key, scans, **`StudentList.xlsx`**). Details: **`AGENTS.md`** and **`pipeline/`**.

```bash
python grade.py "check all multiple choice question answers"
python grade.py "count marks for each student" --folder "Space Physics Unit Test"
python grade.py "check the first 5 students' answers" --dpi 300
```

| Option | Meaning |
|--------|---------|
| `--folder PATH` | Force exam folder |
| `--dpi N` | Override image DPI |
| `--no-cleanup` | Skip cleaning; use `cleaned_scan.pdf` or a `*scan*.pdf` |
| `--output-dir DIR` | Report output directory (default `output/`) |
| `--no-report` | Terminal only; no LaTeX/PDF |

If a ground-truth file is present in that folder, the terminal shows a **colour-coded** summary and the PDF report includes it when LaTeX succeeds.

---

## Ground truth (two places)

Extraction and grading do **not** share the same ground-truth file or code.

| | **`extract_answers.py`** | **`grade.py`** |
|---|--------------------------|----------------|
| **Where the file lives** | Path in **`GROUND_TRUTH_PATH`** (`config.py`) | Inside the **exam folder** (e.g. `ground_truth.txt`, or `Ground Truth ` with a trailing space) |
| **What it is compared to** | Per-page extraction records | **`StudentResult`** from the pipeline |
| **Implementation** | `extraction/ground_truth.py` | `pipeline/ground_truth.py` |

---

## Project layout (high level)

```
autograder.py          # PDF cleaning
extract_answers.py     # Profiled extraction CLI
grade.py               # Generic grading CLI
config.py              # Models, DPI, paths
extraction/            # Profiles, Gemini/Kimi providers, eval, reporting
pipeline/              # Folder discovery, scaffold, grading, GT, PDF report
version.py
AGENTS.md              # Maintainer / agent reference (deeper detail)
```

## Security & privacy

Do not commit **`.env`**, scans, or ground-truth files with student data. Keep sensitive PDFs local; many paths are **gitignored** by design.

## Further reading

**`AGENTS.md`** — dependency versions, naming conventions, eval JSON, LaTeX notes.
