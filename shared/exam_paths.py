"""Paths for per-exam derived artifacts (under ``output/<stem>/`` by default)."""

from __future__ import annotations

from pathlib import Path


def safe_path_stem(stem: str) -> str:
    """Stable directory / filename fragment from a PDF stem (no spaces or slashes)."""
    return stem.replace(" ", "_").replace("/", "_")


def exam_artifact_dir(exam_folder: Path, output_base: str | Path = "output") -> Path:
    """Directory for cleaned scans, scaffold cache, images, and debug PDFs.

    *exam_folder* is the exam input directory (raw PDFs, roster). *stem* is the
    folder name with spaces replaced by underscores.
    """
    stem = exam_folder.name.replace(" ", "_")
    return Path(output_base) / stem


def artifact_scaffold_cache_path(artifact_dir: Path) -> Path:
    return artifact_dir / "scaffold_cache.json"


def legacy_artifact_scaffold_cache_path(artifact_dir: Path) -> Path:
    """Older layout: cache lived under ``scaffolds/`` inside *artifact_dir*."""
    return artifact_dir / "scaffolds" / "scaffold_cache.json"


def artifact_scaffold_boxes_path(artifact_dir: Path) -> Path:
    """Vector-exam PDF with scaffold rectangles drawn (one file per run)."""
    return artifact_dir / "raw_exam_bboxes.pdf"


def extract_answers_output_dir(
    pdf_stem: str, output_base: str | Path = "output"
) -> Path:
    """Directory for one ``extract_answers`` run: ``output/extract_answers/<safe_stem>/``."""
    return Path(output_base) / "extract_answers" / safe_path_stem(pdf_stem)


CLEANED_SCAN_PDF = "cleaned_scan.pdf"


def find_latest_cleaned_scan(
    exam_folder: Path,
    output_base: str | Path = "output",
) -> Path | None:
    """Return the newest ``cleaned_scan.pdf`` among known layouts, or ``None``.

    Searches (all must exist as files to be candidates):

    - ``<output_base>/<safe_stem>/CLEANED_SCAN_PDF`` (flat under exam output stem)
    - ``<output_base>/<safe_stem>/*/CLEANED_SCAN_PDF`` (per-run folders from xscore)
    - ``<exam_folder>/CLEANED_SCAN_PDF`` (legacy next to exam inputs)

    *safe_stem* is ``exam_folder.name`` with spaces replaced by underscores.
    The winner is the path with the largest ``st_mtime``.
    """
    stem = exam_folder.name.replace(" ", "_")
    base = Path(output_base) / stem
    name = CLEANED_SCAN_PDF
    candidates: list[Path] = []

    flat = base / name
    if flat.is_file():
        candidates.append(flat)

    if base.is_dir():
        for p in base.glob(f"*/{name}"):
            if p.is_file():
                candidates.append(p)

    legacy = exam_folder / name
    if legacy.is_file():
        candidates.append(legacy)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_scaffold_cache_file(
    exam_folder: Path, output_base: str | Path = "output"
) -> Path | None:
    """First existing scaffold cache: artifact dir, then legacy locations under *exam_folder*."""
    ad = exam_artifact_dir(exam_folder, output_base)
    for p in (
        artifact_scaffold_cache_path(ad),
        legacy_artifact_scaffold_cache_path(ad),
        exam_folder / "scaffolds" / "scaffold_cache.json",
        exam_folder / "scaffold_cache.json",
    ):
        if p.is_file():
            return p
    return None
