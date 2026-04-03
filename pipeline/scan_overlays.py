"""Write debug PDFs on deskewed scans: reflines overlay and projected scaffold boxes."""

from __future__ import annotations

import json
from pathlib import Path


def write_reflines_debug_pdf(deskewed_pdf: Path, dpi: int) -> Path | None:
    """Draw vertical reflines + anchor crosshairs from the deskew sidecar."""
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
) -> Path | None:
    """Project scaffold bboxes onto *deskewed_pdf* using 4-up anchors + reflines JSON."""
    from pipeline.bbox_projection import find_raw_four_up_pdf, overlay_projected_scaffold_on_scan_pdf
    from pipeline.scaffold import _effective_cache_path, _find_exam_pdf, question_from_dict

    folder = Path(folder)
    deskewed_pdf = Path(deskewed_pdf)

    raw4 = find_raw_four_up_pdf(folder)
    if raw4 is None:
        print(
            "[scan_overlays] No *4up* raw exam PDF — skip projected scaffold overlay "
            "(needs four-up IGCSE header anchors)."
        )
        return None

    cache = _effective_cache_path(folder)
    if cache is None:
        print("[scan_overlays] No scaffold cache — skip projected overlay")
        return None

    try:
        exam_for_scaffold = _find_exam_pdf(folder)
    except FileNotFoundError:
        print("[scan_overlays] No raw exam PDF — skip projected overlay")
        return None

    if not force_layout_mismatch and exam_for_scaffold.resolve() != raw4.resolve():
        print(
            "[scan_overlays] Skip projected overlay: scaffold was built from "
            f"{exam_for_scaffold.name!r} but anchors use {raw4.name!r}. "
            "Use the same 4-up file for scaffold + projection, or run "
            "visualize_scan_overlays.py --force-projected."
        )
        return None

    try:
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        roots = [question_from_dict(q) for q in data["questions"]]
    except Exception as e:
        print(f"[scan_overlays] Could not load scaffold cache: {e}")
        return None

    sidecar = deskewed_pdf.with_name(deskewed_pdf.stem + "_reflines.json")
    if not sidecar.is_file():
        print("[scan_overlays] Missing reflines JSON — skip projected overlay")
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
        )
        return out
    except Exception as e:
        print(f"[scan_overlays] Projected scaffold overlay failed: {e}")
        return None


def write_scan_debug_pdfs_after_deskew(
    folder: Path,
    deskewed_pdf: Path,
    dpi: int,
    *,
    force_projected_mismatch: bool = False,
    write_reflines: bool = True,
    write_projected: bool = True,
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
        )
