#!/usr/bin/env python3
"""
config.py
---------
Configuration for Auto-Grader. Edit values here or set the noted environment
variables. See README.md for full detail.

How to run (from repo root, with venv activated and dependencies installed):

  Setup once:
    python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    # Create .env with GOOGLE_API_KEY or GEMINI_API_KEY (Gemini) and/or KIMI_API_KEY (Kimi).

  Activate the virtual environment:
    source .venv/bin/activate

  Grade an exam folder from a natural-language prompt (uses Kimi only; KIMI_API_KEY):
    python grade.py "check all multiple choice question answers"
    python grade.py "..." --folder "path/to/exam_folder"
    # Optional CLI (also inferable from prompt JSON): --dpi  --folder  --skip-clean-scan
    #   --force-clean-scan  --rescaffold  --through-step N  --no-report

Tunables below apply to extraction/, the other top-level packages, and grade.py
(PIPELINE_*, NAME_*, etc.).
"""

import os
from pathlib import Path

# =============================================================================
# AI Model Configuration
# =============================================================================

# Select which AI model to use for OCR/extraction.
# Options (exact model names - edit this line to change model):
#   - "gemini-3.1-pro-preview"  : Google Gemini 3.1 Pro (highest accuracy)
#   - "gemini-3.0-flash"        : Google Gemini 3.0 Flash (faster, lower accuracy)
#   - "kimi-k2.5"               : Moonshot Kimi K2.5 (OpenAI-compatible API)
#   - "kimi-k2"                 : Moonshot Kimi K2 (alternative name)
#
# To change the model, either:
#   1. Edit the line below, OR
#   2. Set AI_MODEL environment variable (takes precedence)
AI_MODEL = os.getenv("AI_MODEL", "kimi-k2.5")

# Exam layout + prompt + schema (see extraction/profiles/)
EXAM_PROFILE = "igcse_physics"

# =============================================================================
# API Configuration
# =============================================================================

# API keys are read from environment variables:
#   - GOOGLE_API_KEY or GEMINI_API_KEY : For Gemini models
#   - MOONSHOT_API_KEY                  : For Kimi models

# Delay between API calls (seconds). Set to 0 for no delay.
API_CALL_DELAY_S = 0

# Maximum retries for failed API calls
MAX_RETRIES = 3

# Initial backoff time for retries (seconds). Doubles after each failure.
RETRY_BACKOFF_S = 1

# =============================================================================
# Image Processing Configuration
# =============================================================================

# DPI for PDF to image conversion. Higher = better quality but slower.
# 400 DPI is recommended for handwriting recognition.
PDF_DPI = 300

# JPEG quality for image encoding (0-100). Higher = better quality, larger size.
JPEG_QUALITY = 95

# Page images embedded in cleaned_scan.pdf after deskew (see preprocessing/deskew.py).
#   "jpeg" — faster write, smaller file, lossy (uses CLEANED_SCAN_JPEG_QUALITY).
#   "png"  — lossless; still written with parallel encoding (slower than jpeg).
# Change default by editing the getenv second argument, or set CLEANED_SCAN_EMBED_FORMAT.
_csef = os.getenv("CLEANED_SCAN_EMBED_FORMAT", "jpeg").strip().lower()
CLEANED_SCAN_EMBED_FORMAT: str = _csef if _csef in ("jpeg", "png") else "jpeg"
CLEANED_SCAN_JPEG_QUALITY = int(os.getenv("CLEANED_SCAN_JPEG_QUALITY", "95"))

# Fraction of page to crop from top. 0.6 = top 60% of page.
# The answer section is typically in the top half of the page.
CROP_TOP_FRACTION = 0.6

# Image preprocessing enhancement factors
# These help make handwritten marks more visible
PREPROCESS_CONTRAST = 1.5      # Contrast enhancement (1.0 = no change)
PREPROCESS_SHARPNESS = 1.6     # Sharpness enhancement
PREPROCESS_BRIGHTNESS = 1.1    # Brightness adjustment

# =============================================================================
# Ensemble / Multi-Pass Configuration
# =============================================================================

# Enable ensemble voting for improved accuracy.
# When enabled, makes multiple API calls and uses majority voting.
# Slower but potentially more accurate.
USE_ENSEMBLE = False

# Number of API calls per page for ensemble voting.
# Only used when USE_ENSEMBLE is True.
ENSEMBLE_CALLS = 3

# Number of passes for multi-pass extraction (without full ensemble).
# Set to 1 for single-pass (faster), 2+ for multiple passes.
MULTI_PASS_COUNT = 1

# =============================================================================
# Gemini Model Parameters
# =============================================================================

# Temperature controls randomness in model output.
# 0.0 = deterministic, higher = more creative.
GEMINI_TEMPERATURE = 0.1

# Maximum output tokens for Gemini response.
GEMINI_MAX_OUTPUT_TOKENS = 32000

# Enable extended thinking for Gemini 2.5+ models.
# True  → model reasons step-by-step before answering (slower, uses more tokens).
# False → thinking disabled (thinking_budget=0); faster and cheaper.
GEMINI_THINKING = False

# Token budget for Gemini's internal reasoning (only used when GEMINI_THINKING = True).
# Higher values allow deeper reasoning; lower values constrain it.
# Recommended range: 512–8192. Gemini 2.5 Flash cap is 24576.
GEMINI_THINKING_BUDGET = 2048

# =============================================================================
# Kimi Model Parameters
# =============================================================================

# Maximum tokens for Kimi response
KIMI_MAX_TOKENS = 8192

# Enable extended thinking for kimi-k2.x models.
# True  → model reasons step-by-step before answering (slower, uses more tokens).
# False → thinking disabled; faster and cheaper, usually sufficient for OCR.
KIMI_THINKING = False

# =============================================================================
# Paths and File Handling
# =============================================================================

# Default PDF to process (raw scans; see ``Space Physics Unit Test/``)
DEFAULT_PDF = "Space Physics Unit Test/scan 400dpi.pdf"

# Ground truth file path (for accuracy evaluation; repo-relative)
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "Ground Truth "

# =============================================================================
# Generic Pipeline Configuration (grade.py)
# =============================================================================

# DPI used when rendering pages for the main grading pass
PIPELINE_DEFAULT_DPI = 400

# DPI used for quick name-recognition crops (lower = faster, sufficient for names)
NAME_RECOGNITION_DPI = 200

# Fraction of the page height to crop for name detection (top strip only)
NAME_CROP_FRACTION = 0.15

# AI model used by the pipeline (can differ from AI_MODEL used for benchmarking)
PIPELINE_AI_MODEL = "kimi-k2.5"


def resolve_pipeline_ai_model_id() -> str:
    """Return the pipeline model id (``PIPELINE_AI_MODEL`` env overrides the default)."""
    v = os.getenv("PIPELINE_AI_MODEL")
    if v is None or not v.strip():
        return PIPELINE_AI_MODEL
    return v.strip()


def pipeline_ai_model_display_name() -> str:
    """Human-readable label for terminal messages (tracks ``resolve_pipeline_ai_model_id()``)."""
    mid = resolve_pipeline_ai_model_id()
    low = mid.strip().lower()
    if low == "kimi-k2.5":
        return "Kimi K2.5"
    if low == "kimi-k2":
        return "Kimi K2"
    if low.startswith("kimi-"):
        rest = low[5:].replace("-", " ")
        return "Kimi " + rest.upper()
    if low.startswith("gemini-"):
        return "Gemini " + low[7:].replace("-", " ").title()
    return mid
