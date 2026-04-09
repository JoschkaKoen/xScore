"""Clean a class scan PDF (rotate + de-blank + optional deskew) into *artifact_dir*."""

from __future__ import annotations

import shutil
from pathlib import Path

# Phased pipeline artifacts (steps 5–10 scan block in xscore.py README).
SCAN_BLANKS_JSON = "scan_blanks.json"
SCAN_ROTATED_PDF = "scan_rotated.pdf"
CLEANED_SCAN_PDF = "cleaned_scan.pdf"
CLEANED_SCAN_TRANSFORMS_JSON = "cleaned_scan_transforms.json"
PROJECTED_BOXES_SUFFIX = "_projected_boxes.pdf"
REFINED_BOXES_SUFFIX = "_refined_boxes.pdf"
HANDWRITING_RESULTS_JSON = "scan_handwriting_results.json"


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
        "transforms": ad / CLEANED_SCAN_TRANSFORMS_JSON,
        "refined": out.with_name(out.stem + REFINED_BOXES_SUFFIX),
        "hw_results": ad / HANDWRITING_RESULTS_JSON,
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
        detect_blank_page_lists(source_pdf, blank_mean=bm, blank_std=bs)
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
        f"{len(content_page_nums)} content pages · {len(blank_page_nums)} blank"
    )
    return paths["blanks_json"]


def autorotate_phase(
    artifact_dir: Path,
    *,
    output_pdf: Path | None = None,
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
        use_tesseract_rotation=bool(state["use_tesseract_rotation"]),
    )
    return out


def deskew_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
    *,
    input_pdf: Path | None = None,
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
        saved_as=out.name,
    )
    shutil.move(str(tmp_deskew), str(out))
    return out


def detect_page_anchors_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
) -> None:
    """Step 8: fill IGCSE header anchors in the deskew sidecar."""
    del folder  # reserved for API symmetry with other phases
    from preprocessing.deskew import detect_page_anchors_for_cleaned_scan

    paths = _scan_phase_paths(artifact_dir)
    deskewed = paths["cleaned"]
    sidecar = paths["sidecar"]
    if not deskewed.is_file():
        raise FileNotFoundError(f"Missing {deskewed.name} — run deskew first.")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing {sidecar.name} — run deskew first.")
    detect_page_anchors_for_cleaned_scan(deskewed, sidecar, dpi)


def compute_transformation_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
    *,
    force_layout_mismatch: bool = False,
) -> Path | None:
    """Step 9: write ``cleaned_scan_transforms.json`` (4-up ↔ scan similarity per page)."""
    from preprocessing.deskew import resolve_deskew_sidecar
    from scaffold.project_boxes_on_scanned_exam import (
        find_raw_four_up_pdf,
        write_scan_page_transforms_json,
    )
    from scaffold.generate_scaffold import _find_exam_pdf
    from shared.terminal_ui import info_line, ok_line

    paths = _scan_phase_paths(artifact_dir)
    deskewed = paths["cleaned"]
    transforms_path = paths["transforms"]
    if not deskewed.is_file():
        raise FileNotFoundError(f"Missing {deskewed.name} — run deskew first.")

    folder = Path(folder)
    raw4 = find_raw_four_up_pdf(folder)
    if raw4 is None:
        msg = "No *4up* raw exam PDF — skip transforms JSON"
        info_line(msg)
        if transforms_path.is_file():
            transforms_path.unlink()
        return None

    try:
        exam_for_scaffold = _find_exam_pdf(folder)
    except FileNotFoundError:
        info_line("No raw exam PDF — skip transforms JSON")
        if transforms_path.is_file():
            transforms_path.unlink()
        return None

    if not force_layout_mismatch and exam_for_scaffold.resolve() != raw4.resolve():
        msg = (
            "Skip transforms JSON: exam PDF used for scaffold does not match the four-up file."
        )
        info_line(msg)
        if transforms_path.is_file():
            transforms_path.unlink()
        return None

    sidecar = resolve_deskew_sidecar(deskewed)
    if sidecar is None or not sidecar.is_file():
        info_line("Missing anchor sidecar — skip transforms JSON")
        if transforms_path.is_file():
            transforms_path.unlink()
        return None

    if write_scan_page_transforms_json(
        raw4,
        sidecar,
        transforms_path,
        dpi=dpi,
    ):
        ok_line("Done")
        return transforms_path
    if transforms_path.is_file():
        transforms_path.unlink()
    return None


def project_bounding_boxes_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
    *,
    force_layout_mismatch: bool = False,
) -> Path | None:
    """Step 10: draw ``*_projected_boxes.pdf`` using transforms from step 9."""
    from preprocessing.deskew import resolve_deskew_sidecar
    from scaffold.generate_scaffold import _find_exam_pdf, build_scaffold
    from scaffold.project_boxes_on_scanned_exam import (
        find_raw_four_up_pdf,
        overlay_projected_scaffold_from_transforms_json,
        overlay_projected_scaffold_on_scan_pdf,
    )
    from shared.terminal_ui import info_line

    paths = _scan_phase_paths(artifact_dir)
    deskewed = paths["cleaned"]
    transforms_path = paths["transforms"]
    projected = paths["projected"]
    folder = Path(folder)
    ad = artifact_dir

    if not deskewed.is_file():
        raise FileNotFoundError(f"Missing {deskewed.name} — run deskew first.")

    raw4 = find_raw_four_up_pdf(folder)
    if raw4 is None:
        info_line("No *4up* raw exam PDF — skip projected scaffold PDF")
        return None
    try:
        exam_for_scaffold = _find_exam_pdf(folder)
    except FileNotFoundError:
        info_line("No raw exam PDF — skip projected scaffold PDF")
        return None
    if not force_layout_mismatch and exam_for_scaffold.resolve() != raw4.resolve():
        info_line(
            "Skip projected PDF: scaffold exam PDF does not match the four-up file."
        )
        return None

    sidecar = resolve_deskew_sidecar(deskewed)
    if sidecar is None:
        info_line("Missing anchor sidecar — skip projected PDF")
        return None

    try:
        scaffold = build_scaffold(
            folder,
            artifact_dir=ad,
            quiet=True,
        )
        roots = scaffold.questions
    except Exception as e:
        from shared.terminal_ui import warn_line

        warn_line(f"Could not load scaffold for projected overlay: {e}")
        return None

    try:
        if transforms_path.is_file():
            out = overlay_projected_scaffold_from_transforms_json(
                deskewed,
                transforms_path,
                roots,
                projected,
            )
            return out if out is not None and out.is_file() else None
        info_line(
            "No transforms file — drawing projected boxes using sidecar (legacy path)"
        )
        overlay_projected_scaffold_on_scan_pdf(
            deskewed,
            sidecar,
            raw4,
            roots,
            projected,
            dpi=dpi,
        )
        return projected if projected.is_file() else None
    except Exception as e:
        from shared.terminal_ui import warn_line

        warn_line(f"Projected scaffold overlay failed: {e}")
        return None


def refine_bounding_boxes_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
    *,
    pages_to_check: tuple[int, ...] = (0,),
) -> Path | None:
    """Step 11: detect handwriting in yellow margin strips; draw red/green on refined PDF.

    Crops each yellow bbox from the deskewed scan, runs PaddleOCR PPStructure to
    detect handwriting, then draws red (handwriting present) or green (blank) outlines
    on a copy of cleaned_scan_projected_boxes.pdf → cleaned_scan_refined_boxes.pdf.

    Args:
        pages_to_check: Zero-based page indices to analyse. Defaults to page 0 only;
                        pass ``tuple(range(n))`` to extend to all pages later.
    """
    import json

    import fitz
    from rich.progress import Progress

    from scaffold.detect_handwriting import (
        HWResult,
        detect_handwriting_in_rects,
        make_engine,
        overlay_refined_boxes,
    )
    from scaffold.generate_scaffold import _find_exam_pdf, build_scaffold
    from scaffold.project_boxes_on_scanned_exam import (
        compute_yellow_rects_for_page,
        similarity_transform_from_dict,
    )
    from shared.models import flatten_questions
    from shared.terminal_ui import info_line, warn_line

    paths = _scan_phase_paths(artifact_dir)
    projected = paths["projected"]
    refined = paths["refined"]
    hw_json = paths["hw_results"]
    transforms_path = paths["transforms"]
    cleaned = paths["cleaned"]

    if not projected.is_file():
        warn_line("No projected boxes PDF — run step 10 first.")
        return None
    if not transforms_path.is_file():
        warn_line("No transforms JSON — cannot compute yellow rects (run step 9 first).")
        return None
    if not cleaned.is_file():
        warn_line("No cleaned scan PDF — cannot rasterize pages.")
        return None

    try:
        scaffold = build_scaffold(folder, artifact_dir=artifact_dir, quiet=True)
    except Exception as e:
        warn_line(f"Could not load scaffold: {e}")
        return None

    payload = json.loads(transforms_path.read_text())
    dpi_used = int(payload.get("dpi", dpi))
    px_to_pt = 72.0 / dpi_used
    page_entries = payload["pages"]
    all_nodes = list(flatten_questions(scaffold.questions))

    engine = make_engine()
    page_results: dict[int, list[HWResult]] = {}

    with Progress() as progress:
        task = progress.add_task(
            "Checking yellow strips for handwriting …", total=len(pages_to_check)
        )
        doc = fitz.open(str(cleaned))
        for page_idx in pages_to_check:
            if page_idx >= len(page_entries) or page_idx >= len(doc):
                warn_line(f"Page {page_idx} out of range — skipping.")
                progress.advance(task)
                continue
            pe = page_entries[page_idx]
            top_tf = similarity_transform_from_dict(pe["top"])
            bot_tf = similarity_transform_from_dict(pe["bot"])
            page = doc[page_idx]
            rects = compute_yellow_rects_for_page(
                page, all_nodes, top_tf, bot_tf, px_to_pt=px_to_pt
            )
            hw_results = detect_handwriting_in_rects(
                cleaned, page_idx, rects, dpi_used, engine
            )
            page_results[page_idx] = hw_results
            progress.advance(task)
        doc.close()

    serialisable = [
        {
            "page": page_idx,
            "rect": [hw.rect.x0, hw.rect.y0, hw.rect.x1, hw.rect.y1],
            "has_handwriting": hw.has_handwriting,
        }
        for page_idx, results in page_results.items()
        for hw in results
    ]
    hw_json.write_text(json.dumps(serialisable, indent=2))
    info_line(f"Saved {hw_json.name}")

    overlay_refined_boxes(projected, refined, page_results)
    return refined


def calculate_transformation_phase(
    folder: Path,
    artifact_dir: Path,
    dpi: int,
) -> Path | None:
    """Run steps 9–10 only (transforms JSON + projected boxes PDF).

    Prefer calling :func:`compute_transformation_phase` and
    :func:`project_bounding_boxes_phase` separately from the CLI.
    """
    compute_transformation_phase(folder, artifact_dir, dpi)
    return project_bounding_boxes_phase(folder, artifact_dir, dpi)


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
    detect anchors → compute transforms → projected overlay). For partial runs, use the
    individual ``*_phase`` functions
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
        autorotate_phase(ad)
        deskew_phase(folder, ad, dpi)
        detect_page_anchors_phase(folder, ad, dpi)
        compute_transformation_phase(folder, ad, dpi)
        project_bounding_boxes_phase(folder, ad, dpi)
    else:
        autorotate_phase(ad, output_pdf=output)

    return output
