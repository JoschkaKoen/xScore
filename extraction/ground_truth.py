"""Ground truth file loading, fuzzy name match, per-student accuracy."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from config import GROUND_TRUTH_PATH


def load_ground_truth(gt_path: Path | None = None) -> dict[str, list[str]]:
    """Load ground truth answers from file.

    Returns dict mapping student_name -> answer columns in profile order.
    """
    path = gt_path if gt_path is not None else GROUND_TRUTH_PATH
    if not path.exists():
        return {}

    gt_data: dict[str, list[str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header line
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 7:  # name + 6 answers
                name = parts[0]
                answers = parts[1:7]
                gt_data[name] = answers
    except OSError as e:
        from shared.terminal_ui import warn_line

        warn_line(f"Could not load ground truth from {path}: {e}")

    return gt_data


def fuzzy_match_name(extracted_name: str, gt_names: list[str]) -> str | None:
    """Find the best matching ground truth name using fuzzy matching."""
    if not extracted_name or extracted_name in ("UNKNOWN", "EXTRACTION_ERROR"):
        return None

    extracted_lower = extracted_name.lower().strip()

    for gt_name in gt_names:
        if gt_name.lower() == extracted_lower:
            return gt_name

    for gt_name in gt_names:
        gt_lower = gt_name.lower()
        if extracted_lower in gt_lower or gt_lower in extracted_lower:
            return gt_name

    best_match = None
    best_ratio = 0.0
    for gt_name in gt_names:
        ratio = SequenceMatcher(None, extracted_lower, gt_name.lower()).ratio()
        if ratio > best_ratio and ratio >= 0.6:
            best_ratio = ratio
            best_match = gt_name

    return best_match


def calculate_student_accuracy(
    extracted: dict, gt_answers: list[str], answer_fields: list[str]
) -> float:
    """Calculate accuracy percentage for a single student."""
    correct = 0
    total = len(answer_fields)

    for i, field in enumerate(answer_fields):
        extracted_val = extracted.get(field, "?").upper().strip()
        gt_val = gt_answers[i].upper().strip() if i < len(gt_answers) else ""

        if extracted_val == gt_val and extracted_val not in ("", "?"):
            correct += 1

    return (correct / total) * 100 if total > 0 else 0.0
