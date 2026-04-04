"""Tests for OCR benchmark helpers and ``find_latest_cleaned_scan``."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from shared.exam_paths import find_latest_cleaned_scan


def test_find_latest_cleaned_scan_picks_newer_mtime(tmp_path: Path) -> None:
    exam = tmp_path / "My Exam"
    exam.mkdir()
    out_root = tmp_path / "output"
    stem_dir = out_root / "My_Exam"
    run_a = stem_dir / "run_a"
    run_b = stem_dir / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    older = run_a / "cleaned_scan.pdf"
    newer = run_b / "cleaned_scan.pdf"
    older.write_text("a")
    time.sleep(0.02)
    newer.write_text("b")

    found = find_latest_cleaned_scan(exam, output_base=out_root)
    assert found is not None
    assert found.resolve() == newer.resolve()


def test_find_latest_cleaned_scan_legacy_in_exam_folder(tmp_path: Path) -> None:
    exam = tmp_path / "Exam"
    exam.mkdir()
    legacy = exam / "cleaned_scan.pdf"
    legacy.write_text("x")

    found = find_latest_cleaned_scan(exam, output_base=tmp_path / "output")
    assert found == legacy


def test_find_latest_cleaned_scan_none_when_missing(tmp_path: Path) -> None:
    exam = tmp_path / "Empty"
    exam.mkdir()
    assert find_latest_cleaned_scan(exam, output_base=tmp_path / "out") is None


def test_pts_to_px() -> None:
    from scripts.ocr_name_benchmark import pts_to_px

    assert pts_to_px(72, 72) == 72
    assert pts_to_px(840, 300) == 3500
