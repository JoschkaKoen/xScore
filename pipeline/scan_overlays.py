"""Write debug PDFs on deskewed scans (projected scaffold boxes).

The reflines overlay PDF writer is kept for optional manual use; the pipeline does not
generate ``*_reflines_overlay.pdf`` by default (see :func:`write_reflines_debug_pdf`).
"""

from __future__ import annotations

from pathlib import Path


def write_reflines_debug_pdf(deskewed_pdf: Path, dpi: int) -> Path | None:
    """Draw vertical reflines + anchor crosshairs from the deskew sidecar.

    Not invoked from :func:`write_scan_debug_pdfs_after_deskew` by default; call this
    explicitly (or use ``scripts/visualize_scan_overlays.py --reflines-overlay``) if you need
    the PDF for debugging.
    """
    deskewed_pdf = Path(deskewed_pdf)
    sidecar = deskewed_pdf.with_name(deskewed_pdf.stem + "_reflines.json")
    if not sidecar.is_file():
        print(f"[scan_overlays] No reflines sidecar ({sidecar.name}) — skip reflines overlay")
        return None
    out = deskewed_pdf.with_name(deskewed_pdf.stem + "_reflines_overlay.pdf")
    try:
        from pipeline.scan_deskew import overlay_reflines_on_pdf

        overlay_reflines_on_pdf(deskewed_pdf, sidecar, out, dpi=dpi)
        return out
    except Exception as e:
        print(f"[scan_overlays] Reflines overlay failed: {e}")
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
    Pass the same directory as ``grade.py`` / ``cleanup_pdf`` so overlays match the current run.
    """
    from pipeline.bbox_projection import find_raw_four_up_pdf, overlay_projected_scaffold_on_scan_pdf
    from pipeline.exam_paths import exam_artifact_dir
    from pipeline.scaffold import _find_exam_pdf, build_scaffold
    from pipeline.terminal_ui import info_line, warn_line

    folder = Path(folder)
    deskewed_pdf = Path(deskewed_pdf)

    raw4 = find_raw_four_up_pdf(folder)
    if raw4 is None:
        msg = (
            "No *4up* raw exam PDF — skip projected overlay (needs four-up IGCSE anchors)."
        )
        (warn_line if verbose else info_line)(f"[scan_overlays] {msg}")
        return None

    try:
        exam_for_scaffold = _find_exam_pdf(folder)
    except FileNotFoundError:
        (warn_line if verbose else info_line)("[scan_overlays] No raw exam PDF — skip projected overlay")
        return None

    if not force_layout_mismatch and exam_for_scaffold.resolve() != raw4.resolve():
        msg = (
            "Skip projected overlay: scaffold from "
            f"{exam_for_scaffold.name!r} but anchors use {raw4.name!r}. "
            "Use the same 4-up file or scripts/visualize_scan_overlays.py --force-projected."
        )
        (warn_line if verbose else info_line)(f"[scan_overlays] {msg}")
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
        warn_line(f"[scan_overlays] Could not load scaffold: {e}")
        return None

    sidecar = deskewed_pdf.with_name(deskewed_pdf.stem + "_reflines.json")
    if not sidecar.is_file():
        (warn_line if verbose else info_line)(
            "[scan_overlays] Missing reflines JSON — skip projected overlay"
        )
        return None

    out = deskewed_pdf.with_name(deskewed_pdf.stem + "_projected_boxes.pdf")
    try:
        overlay_projected_scaffold_on_scan_pdf(
            deskewed_pdf,
            sidecar,
            raw4,
            roots,
            out,
            dpi=dpi,
            verbose=verbose,
        )
        return out
    except Exception as e:
        warn_line(f"[scan_overlays] Projected scaffold overlay failed: {e}")
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
