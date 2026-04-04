#!/usr/bin/env bash
# Optional PaddleOCR + EasyOCR stack in an isolated venv (paddle_env/).
# xScore grading still uses .venv — see README.
#
# Requires Python 3.13 (max supported by Paddle wheels; 3.14+ has no wheel).
# Looks for python3.13 on PATH, then falls back to python3.12.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

echo ""
echo "=== PaddleOCR + EasyOCR setup (experimental) ==="
echo "Repo: ${REPO_ROOT}"
echo ""

if ! command -v brew &>/dev/null; then
  echo "Homebrew not found. Install from https://brew.sh/ then run this script again."
  exit 1
fi

# Pick Python ≤3.13 — Paddle has no wheel for 3.14+
if command -v python3.13 &>/dev/null; then
  PYTHON_BIN="$(command -v python3.13)"
elif command -v python3.12 &>/dev/null; then
  PYTHON_BIN="$(command -v python3.12)"
else
  echo "Python 3.12 or 3.13 is required (PaddlePaddle has no wheel for 3.14+)."
  echo "Install with: brew install python@3.13"
  exit 1
fi
echo "Using Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version))"

echo "--- Homebrew: cmake, pkg-config, opencv ---"
brew install cmake pkg-config opencv

VENV="${REPO_ROOT}/paddle_env"
if [[ "${FORCE:-0}" == "1" ]] && [[ -d "${VENV}" ]]; then
  echo "--- FORCE=1: removing existing paddle_env ---"
  rm -rf "${VENV}"
fi

if [[ ! -d "${VENV}" ]]; then
  echo "--- Creating paddle_env (${PYTHON_BIN}) ---"
  "${PYTHON_BIN}" -m venv "${VENV}"
else
  echo "--- Using existing paddle_env (set FORCE=1 to recreate) ---"
fi

PY="${VENV}/bin/python"

echo "--- Upgrading pip ---"
"${PY}" -m pip install --upgrade pip

# Bump version when Paddle docs change: https://www.paddlepaddle.org.cn/
echo "--- Step A: PaddlePaddle (official CPU index) ---"
"${PY}" -m pip install paddlepaddle==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

echo "--- Step B: paddleocr (PyPI) ---"
"${PY}" -m pip install paddleocr

echo "--- Step C: easyocr (PyPI; pulls PyTorch — large download) ---"
"${PY}" -m pip install easyocr

echo "--- Step D: scripts/ocr_name_benchmark.py (Rich, PyMuPDF, OpenCV, Tesseract, roster xlsx) ---"
"${PY}" -m pip install \
  "rich>=13.7.0" \
  "pymupdf>=1.24.0" \
  "pytesseract>=0.3.13" \
  "openpyxl>=3.1.0" \
  "opencv-python-headless>=4.9.0" \
  "Pillow>=10.0.0"

echo "--- Checking ocr_name_benchmark imports ---"
"${PY}" - <<'PY'
import rich
import cv2
import fitz  # pymupdf
import openpyxl
import pytesseract
from PIL import Image
print("ocr_name_benchmark dependencies OK")
PY

echo ""
echo "--- Verifying (first run may download model weights — large, normal) ---"
"${PY}" - <<'PY'
import paddle
paddle.utils.run_check()
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True, lang="en")
print("PaddleOCR ready")
import easyocr
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR ready")
PY

echo ""
echo "=== Done ==="
echo "Activate this stack:  source paddle_env/bin/activate"
echo "Grade with xScore:    use .venv (python3 -m venv .venv && pip install -r requirements.txt)"
echo ""
