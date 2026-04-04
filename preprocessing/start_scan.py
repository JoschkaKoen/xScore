"""Clean a class scan PDF (rotate + de-blank + optional deskew) into *artifact_dir*."""

from __future__ import annotations

import shutil
from pathlib import Path

# Phased pipeline artifacts (steps 5–8 in xscore.py README).
SCAN_BLANKS_JSON = "scan_blanks.json"
SCAN_ROTATED_PDF = "scan_rotated.pdf"
CLEANED_SCAN_PDF = "cleaned_scan.pdf"
PROJECTED_BOXES_SUFFIX = "_projected_boxes.pdf"


def _scan_phase_paths(artifact_dir: Path) -> dict[str, Path]:
    ad = artifact_dir
    out = ad / CLEANED_SCAN_PDF
    return {
        "blanks_json": ad / SCAN_BLANKS_JSON,
        "rotated": ad / SCAN_ROTATED_PDF,
        "cleaned": out,
        "sidecar": out.with_name(f"{out.stem}_anchors.json"),
        "sidecar_legacy": out.with_name(f"{out.stem}_reflines.json"),
        "deskew_tmp": ad / f"{out.stem}_deskew_tmp{out.suffix}",
        "projected": out.with_name(out.stem + PROJECTED_BOXES_SUFFIX),
    }


def _remove_scan_pipeline_outputs(artifact_dir: Path, *, include_projected: bool = True) -> None:
    """Delete intermediate and final scan outputs under *artifact_dir* (force-clean)."""
    p = _scan_phase_paths(artifact_dir)
    for key, path in p.items():
        if key == "projected" and not include_projected:
            continue
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def find_source_scan_match(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
) -> Path:
    """Pick the class-scan PDF under *folder* (same rules as :func:`cleanup_pdf`)."""
    output = artifact_dir / CLEANED_SCAN_PDF
    legacy_out = folder / CLEANED_SCAN_PDF
    scans = [
        f
        for f in folder.glob("*.pdf")
        if "scan" in f.name.lower()
        and f.resolve() not in {output.resolve(), legacy_out.resolve()}
    ]
    if not scans:
        raise FileNotFoundError(f"No scan PDF found in {folder}")
    return next(
        (s for s in scans if str(dpi) in s.stem),
        sorted(scans, key=lambda p: p.name.lower())[0],
    )


def detect_blank_pages_phase(
    source_pdf: Path,
    artifact_dir: Path,
    *,
    analysis_dpi: int,
    force_clean_scan: bool = False,
    blank_mean: float | None = None,
    blank_std: float | None = None,
) -> Path:
    """Step 5: write ``scan_blanks.json`` with blank/content lists and render sizes."""
    from config import SCAN_USE_TESSERACT_ROTATION
    from preprocessing.remove_blanks_autorotate import (
        BLANK_MEAN_THRESHOLD,
        BLANK_STD_THRESHOLD,
        detect_blank_page_lists,
        scan_blanks_state_to_json,
    )
    from shared.terminal_ui import ok_line

    paths = _scan_phase_paths(artifact_dir)
    if force_clean_scan:
        _remove_scan_pipeline_outputs(artifact_dir)

    bm = blank_mean if blank_mean is not None else BLANK_MEAN_THRESHOLD
    bs = blank_std if blank_std is not None else BLANK_STD_THRESHOLD

    total_pages, content_page_nums, blank_page_nums, page_render_sizes = (
        detect_blank_page_lists(source_pdf, blank_mean=bm, blank_std=bs, verbose=False)
    )
    if not content_page_nums:
        raise RuntimeError("All scan pages classified as blank — nothing to process.")

    body = scan_blanks_state_to_json(
        source_pdf=source_pdf,
        total_pages=total_pages,
        content_page_nums=content_page_nums,
        blank_page_nums=blank_page_nums,
        page_render_sizes=page_render_sizes,
        blank_mean=bm,
        blank_std=bs,
        use_tesseract_rotation=SCAN_USE_TESSERACT_ROTATION,
        analysis_dpi=analysis_dpi,
    )
    paths["blanks_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["blanks_json"].write_text(body, encoding="utf-8")
    ok_line(
        f"{len(content_page_nums)} content pages · {len(blank_page_nums)} blank "
        f"(saved {SCAN_BLANKS_JSON})"
    )
    return paths["blanks_json"]


def autorotate_phase(
    artifact_dir: Path,
    *,
    output_pdf: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Step 6: read ``scan_blanks.json``, write rotated PDF (blanks dropped)."""
    from preprocessing.remove_blanks_autorotate import (
        scan_blanks_state_from_json,
        write_rotated_pdf_after_blanks,
    )

    paths = _scan_phase_paths(artifact_dir)
    blanks_path = paths["blanks_json"]
    if not blanks_path.is_file():
        raise FileNotFoundError(f"Missing {blanks_path.name} — run blank detection first.")
    state = scan_blanks_state_from_json(blanks_path.read_text(encoding="utf-8"))
    source = Path(state["source_pdf"])
    if not source.is_file():
        raise FileNotFoundError(f"Source scan missing: {source}")

    out = output_pdf if output_pdf is not None else paths["rotated"]
    write_rotated_pdf_after_blanks(
        source,
        out,
        total_pages=int(state["total_pages"]),
        content_page_nums=list(state["content_page_nums"]),
        blank_page_nums=list(state["blank_page_nums"]),
        page_render_sizes=state["page_render_sizes"],
        analysis_dpi=int(state["analysis_dpi"]),
        verbose=verbose,
        use_tesseract_rotation=bool(state["use_tesseract_rotation"]),
    )
    return out


def deskew_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
    *,
    input_pdf: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Step 7: deskew ``scan_rotated.pdf`` (or *input_pdf*) into ``cleaned_scan.pdf``."""
    from preprocessing.deskew import deskew_pdf_raster

    paths = _scan_phase_paths(artifact_dir)
    inp = input_pdf if input_pdf is not None else paths["rotated"]
    if not inp.is_file():
        raise FileNotFoundError(f"Missing rotated scan: {inp}")

    out = paths["cleaned"]
    tmp_deskew = paths["deskew_tmp"]
    deskew_pdf_raster(
        input_pdf=inp,
        output_pdf=tmp_deskew,
        dpi=dpi,
        reflines_sidecar=out.with_name(f"{out.stem}_anchors.json"),
        verbose=verbose,
        saved_as=out.name,
    )
    shutil.move(str(tmp_deskew), str(out))
    return out


def calculate_transformation_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
    *,
    verbose: bool = False,
) -> Path | None:
    """Step 8: project scaffold boxes onto deskewed scan (exam→scan transform debug PDF)."""
    from preprocessing.draw_scaffold_bounding_boxes import write_scan_debug_pdfs_after_deskew

    paths = _scan_phase_paths(artifact_dir)
    deskewed = paths["cleaned"]
    if not deskewed.is_file():
        raise FileNotFoundError(f"Missing {deskewed.name} — run deskew first.")

    write_scan_debug_pdfs_after_deskew(
        folder,
        deskewed,
        dpi,
        verbose=verbose,
        artifact_dir=artifact_dir,
    )
    projected = paths["projected"]
    return projected if projected.is_file() else None


def cleanup_pdf(
    folder: Path,
    dpi: int = 300,
    deskew: bool = True,
    *,
    artifact_dir: Path | None = None,
    output_base: str | Path = "output",
    force_clean_scan: bool = False,
) -> Path:
    """Clean the scan PDF found in *folder*; write ``cleaned_scan.pdf`` under *artifact_dir*.

    Runs the full phased pipeline internally (detect blanks → autorotate → deskew →
    transformation overlay). For partial runs, use the individual ``*_phase`` functions
    from the ``xscore.py`` CLI.
    """
    from shared.exam_paths import exam_artifact_dir

    ad = artifact_dir or exam_artifact_dir(folder, output_base)
    ad.mkdir(parents=True, exist_ok=True)

    output = ad / CLEANED_SCAN_PDF
    legacy_out = folder / CLEANED_SCAN_PDF

    match = find_source_scan_match(folder, ad, dpi)

    from shared.terminal_ui import tool_line

    sidecar = output.with_name(f"{output.stem}_anchors.json")
    sidecar_legacy_reflines = output.with_name(f"{output.stem}_reflines.json")
    legacy_side = legacy_out.with_name(f"{legacy_out.stem}_anchors.json")
    legacy_side_reflines = legacy_out.with_name(f"{legacy_out.stem}_reflines.json")

    if force_clean_scan:
        _removed = False
        for p in (
            output,
            sidecar,
            sidecar_legacy_reflines,
            legacy_out,
            legacy_side,
            legacy_side_reflines,
        ):
            if p.exists():
                p.unlink()
                _removed = True
        _remove_scan_pipeline_outputs(ad)
        if _removed:
            tool_line("start_scan", "Removed previous cleaned output (force).")

    if not force_clean_scan and output.exists() and output.stat().st_mtime >= match.stat().st_mtime:
        tool_line("start_scan", "Using cached cleaned scan.")
        return output

    if not force_clean_scan and legacy_out.exists() and legacy_out.stat().st_mtime >= match.stat().st_mtime:
        tool_line("start_scan", "Moving old cleaned scan into this run …")
        shutil.copy2(legacy_out, output)
        if legacy_side.is_file():
            shutil.copy2(legacy_side, sidecar)
        elif legacy_side_reflines.is_file():
            shutil.copy2(legacy_side_reflines, sidecar)
        try:
            legacy_out.unlink()
        except OSError:
            pass
        for leg in (legacy_side, legacy_side_reflines):
            if leg.is_file():
                try:
                    leg.unlink()
                except OSError:
                    pass
        return output

    tool_line("start_scan", "Phased scan prep …")
    detect_blank_pages_phase(
        match,
        ad,
        analysis_dpi=dpi,
        force_clean_scan=False,
    )
    if deskew:
        autorotate_phase(ad, verbose=False)
        deskew_phase(folder, ad, dpi, verbose=False)
        calculate_transformation_phase(folder, ad, dpi, verbose=False)
    else:
        autorotate_phase(ad, output_pdf=output, verbose=False)

    return output
