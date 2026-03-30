# Auto-Grader

CLI tool that cleans scanned exam PDFs by:

1. **Auto-rotating** pages to upright orientation using Tesseract OSD  
2. **Dropping** near-blank / white pages  
3. **Writing** a new PDF (pages rendered as JPEG and assembled with `img2pdf`)

## Requirements

- **Python** 3.10 or newer  
- **System binaries** (not installed via pip):
  - [Poppler](https://poppler.freedesktop.org/) — used by `pdf2image` (`pdftoppm`, etc.)
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — used for orientation detection

### Installing system dependencies

- **macOS (Homebrew):**  
  `brew install poppler tesseract`

- **Debian / Ubuntu:**  
  `sudo apt install poppler-utils tesseract-ocr`

## Setup

Use a virtual environment (recommended on macOS Homebrew Python, which blocks global `pip` installs):

```bash
cd /path/to/Auto-Grader
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python autograder.py input.pdf output.pdf
```

### Optional arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--dpi` | `150` | Rasterization DPI (higher = sharper, slower) |
| `--blank-threshold` | `250` | Grayscale mean ≥ this → candidate blank (0–255) |
| `--blank-std` | `5` | Grayscale std ≤ this → candidate blank |

Example:

```bash
python autograder.py scan.pdf cleaned.pdf --dpi 200 --blank-threshold 248 --blank-std 6
```

## Project layout

- `autograder.py` — main script  
- `requirements.txt` — Python dependencies  
