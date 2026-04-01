"""Wrapper around autograder.process_pdf() — cleans a scanned exam PDF."""

from __future__ import annotations

from pathlib import Path


def cleanup_pdf(folder: Path, dpi: int = 300) -> Path:
    """Clean the scan PDF in *folder* and return the path to the output PDF.

    Looks for a file whose name contains "scan" (case-insensitive).
    Output: ``{folder}/cleaned_scan.pdf``.
    Skips processing if the output already exists and is newer than the source.

    Raises ``FileNotFoundError`` if no scan PDF is found.
    """
    # Late import so the pipeline package doesn't force autograder deps at module load
    from autograder import process_pdf  # type: ignore[import]

    scans = [
        f for f in folder.glob("*.pdf")
        if "scan" in f.name.lower()
    ]
    if not scans:
        raise FileNotFoundError(f"No scan PDF found in {folder}")

    # Prefer the scan whose DPI label (if any) matches the requested DPI,
    # falling back to the first one found.
    match = next(
        (s for s in scans if str(dpi) in s.stem),
        scans[0],
    )

    output = folder / "cleaned_scan.pdf"

    if output.exists() and output.stat().st_mtime >= match.stat().st_mtime:
        print(f"[pdf_cleanup] Using cached cleaned scan: {output}")
        return output

    print(f"[pdf_cleanup] Cleaning {match.name} → {output.name} (DPI {dpi}) …")
    process_pdf(
        input_path=str(match),
        output_path=str(output),
        analysis_dpi=dpi,
    )
    return output
