"""Write debug PDFs on deskewed scans (projected scaffold boxes).

The reflines overlay PDF writer is kept for optional manual use; the pipeline does not
generate ``*_reflines_overlay.pdf`` by default (see :func:`write_reflines_debug_pdf`).
"""

from __future__ import annotations

from pathlib import Path


def write_reflines_debug_pdf(deskewed_pdf: Path, dpi: int) -> Path | None:
    """Draw vertical reflines + anchor crosshairs from the deskew sidecar.

    Not invoked from :func:`write_scan_debug_pdfs_after_deskew` by default; call this
    explicitly if you need
    the PDF for debugging.
    """
    deskewed_pdf = Path(deskewed_pdf)
    from preprocessing.deskew import overlay_reflines_on_pdf, resolve_deskew_sidecar

    sidecar = resolve_deskew_sidecar(deskewed_pdf)
    if sidecar is None:
        from shared.terminal_ui import info_line

        info_line(
            "No anchor sidecar (*_anchors.json or legacy *_reflines.json) — skip reflines overlay"
        )
        return None
    out = deskewed_pdf.with_name(deskewed_pdf.stem + "_reflines_overlay.pdf")
    try:
        overlay_reflines_on_pdf(deskewed_pdf, sidecar, out, dpi=dpi)
        return out
    except Exception as e:
        from shared.terminal_ui import warn_line

        warn_line(f"Reflines overlay failed: {e}")
        return None


def write_projected_scaffold_debug_pdf(
    folder: Path,
    deskewed_pdf: Path,
    dpi: int,
    *,
    force_layout_mismatch: bool = False,
    verbose: bool = True,
    artifact_dir: Path | None = None,
) -> Path | None:
    """Project scaffold bboxes onto *deskewed_pdf* using 4-up anchors + reflines JSON.

    *artifact_dir*: scaffold cache location (defaults to :func:`exam_artifact_dir` for the exam).
    Pass the same directory as ``xscore.py`` / ``start_scan.cleanup_pdf`` so overlays match the current run.
    """
    from preprocessing.deskew import resolve_deskew_sidecar
    from scaffold.project_boxes_on_scanned_exam import (
        find_raw_four_up_pdf,
        overlay_projected_scaffold_from_transforms_json,
        overlay_projected_scaffold_on_scan_pdf,
        write_scan_page_transforms_json,
    )
    from shared.exam_paths import exam_artifact_dir
    from scaffold.generate_scaffold import _find_exam_pdf, build_scaffold
    from shared.terminal_ui import info_line, warn_line

    folder = Path(folder)
    deskewed_pdf = Path(deskewed_pdf)

    raw4 = find_raw_four_up_pdf(folder)
    if raw4 is None:
        msg = (
            "No *4up* raw exam PDF — skip projected overlay (needs four-up IGCSE anchors)."
        )
        (warn_line if verbose else info_line)(msg)
        return None

    try:
        exam_for_scaffold = _find_exam_pdf(folder)
    except FileNotFoundError:
        (warn_line if verbose else info_line)("No raw exam PDF — skip projected overlay")
        return None

    if not force_layout_mismatch and exam_for_scaffold.resolve() != raw4.resolve():
        msg = (
            "Skip projected overlay: the exam PDF used for the scaffold does not match "
            "the four-up scan used for anchors. Use the same file for both, or pass "
            "force_layout_mismatch."
        )
        (warn_line if verbose else info_line)(msg)
        return None

    ad = artifact_dir if artifact_dir is not None else exam_artifact_dir(folder)
    try:
        scaffold = build_scaffold(
            folder,
            artifact_dir=ad,
            quiet=not verbose,
        )
        roots = scaffold.questions
    except Exception as e:
        warn_line(f"Could not load scaffold for projected overlay: {e}")
        return None

    sidecar = resolve_deskew_sidecar(deskewed_pdf)
    if sidecar is None:
        (warn_line if verbose else info_line)(
            "Missing anchor sidecar (*_anchors.json or legacy *_reflines.json) — skip projected overlay"
        )
        return None

    out = deskewed_pdf.with_name(deskewed_pdf.stem + "_projected_boxes.pdf")
    transforms_path = deskewed_pdf.with_name(deskewed_pdf.stem + "_transforms.json")
    try:
        if write_scan_page_transforms_json(
            raw4,
            sidecar,
            transforms_path,
            dpi=dpi,
            verbose=verbose,
        ):
            overlay_projected_scaffold_from_transforms_json(
                deskewed_pdf,
                transforms_path,
                roots,
                out,
                verbose=verbose,
            )
        else:
            overlay_projected_scaffold_on_scan_pdf(
                deskewed_pdf,
                sidecar,
                raw4,
                roots,
                out,
                dpi=dpi,
                verbose=verbose,
            )
        return out if out.is_file() else None
    except Exception as e:
        warn_line(f"Projected scaffold overlay failed: {e}")
        return None


def write_scan_debug_pdfs_after_deskew(
    folder: Path,
    deskewed_pdf: Path,
    dpi: int,
    *,
    force_projected_mismatch: bool = False,
    write_reflines: bool = False,
    write_projected: bool = True,
    verbose: bool = False,
    artifact_dir: Path | None = None,
) -> None:
    """After a successful deskew, write optional debug PDFs next to *deskewed_pdf*."""
    if write_reflines:
        write_reflines_debug_pdf(deskewed_pdf, dpi)
    if write_projected:
        write_projected_scaffold_debug_pdf(
            folder,
            deskewed_pdf,
            dpi,
            force_layout_mismatch=force_projected_mismatch,
            verbose=verbose,
            artifact_dir=artifact_dir,
        )
