"""Clean a class scan PDF (rotate + de-blank + optional deskew) into *artifact_dir*."""

from __future__ import annotations

import shutil
from pathlib import Path


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

    Looks for a file whose name contains "scan" (case-insensitive) in *folder*,
    excluding any ``cleaned_scan.pdf`` sitting next to the sources.
    Default *artifact_dir* is ``output/<exam_stem>/`` (see :func:`pipeline.exam_paths.exam_artifact_dir`).

    If a valid cached ``cleaned_scan.pdf`` exists only under the exam folder (legacy),
    it is copied into *artifact_dir* once.

    Skips processing if the output already exists and is newer than the source,
    unless *force_clean_scan* is true (output and matching reflines sidecar are removed first).

    Pass 1: blank page removal (72 DPI)
    Pass 2: OSD 90-degree rotation (pikepdf lossless)
    Pass 3: per-half fine deskew via projection variance (rasterised at *dpi*)
            — only when ``deskew=True`` (default).

    Raises ``FileNotFoundError`` if no scan PDF is found.
    """
    from pipeline.exam_paths import exam_artifact_dir

    ad = artifact_dir or exam_artifact_dir(folder, output_base)
    ad.mkdir(parents=True, exist_ok=True)

    from pipeline.scan_preprocess import process_pdf

    output = ad / "cleaned_scan.pdf"
    legacy_out = folder / "cleaned_scan.pdf"

    scans = [
        f
        for f in folder.glob("*.pdf")
        if "scan" in f.name.lower()
        and f.resolve() not in {output.resolve(), legacy_out.resolve()}
    ]
    if not scans:
        raise FileNotFoundError(f"No scan PDF found in {folder}")

    match = next(
        (s for s in scans if str(dpi) in s.stem),
        sorted(scans, key=lambda p: p.name.lower())[0],
    )

    from pipeline.terminal_ui import tool_line

    sidecar = output.with_name(f"{output.stem}_reflines.json")
    legacy_side = legacy_out.with_name(f"{legacy_out.stem}_reflines.json")

    if force_clean_scan:
        for p in (output, sidecar, legacy_out, legacy_side):
            if p.exists():
                p.unlink()
                tool_line("pdf_cleanup", f"--force-clean-scan: removed {p.name}")

    if not force_clean_scan and output.exists() and output.stat().st_mtime >= match.stat().st_mtime:
        tool_line("pdf_cleanup", f"Using cached cleaned scan: {output}")
        return output

    if not force_clean_scan and legacy_out.exists() and legacy_out.stat().st_mtime >= match.stat().st_mtime:
        tool_line("pdf_cleanup", f"Migrating legacy {legacy_out.name} → {output} …")
        shutil.copy2(legacy_out, output)
        if legacy_side.is_file():
            shutil.copy2(legacy_side, sidecar)
        try:
            legacy_out.unlink()
        except OSError:
            pass
        if legacy_side.is_file():
            try:
                legacy_side.unlink()
            except OSError:
                pass
        return output

    tool_line("pdf_cleanup", f"Cleaning {match.name} → {output.name} (DPI {dpi}) …")
    process_pdf(
        input_path=str(match),
        output_path=str(output),
        analysis_dpi=dpi,
        verbose=False,
    )

    if deskew:
        from pipeline.scan_deskew import deskew_pdf_raster  # type: ignore[import]

        tmp_deskew = output.parent / f"{output.stem}_deskew_tmp{output.suffix}"
        deskew_pdf_raster(
            input_pdf=output,
            output_pdf=tmp_deskew,
            dpi=dpi,
            reflines_sidecar=output.with_name(f"{output.stem}_reflines.json"),
            verbose=False,
        )
        shutil.move(str(tmp_deskew), str(output))
        from pipeline.scan_overlays import write_scan_debug_pdfs_after_deskew

        write_scan_debug_pdfs_after_deskew(
            folder, output, dpi, verbose=False, artifact_dir=ad,
        )

    return output
