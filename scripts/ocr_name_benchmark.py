#!/usr/bin/env python3
"""
Compare Tesseract, EasyOCR, and PaddleOCR on the same name strip from a cleaned class scan.

Uses a fixed PDF-point crop (default 0,0 → 840,125 pt), not the production Kimi top-15% crop.
Match scores use ``fuzzy_match_name`` (same helper as grading).

Optional stacks (EasyOCR / Paddle): install via ``bash scripts/install_paddleocr.sh``,
then ``source paddle_env/bin/activate`` and run this script from that environment.

System: ``brew install tesseract`` (see main README).

Usage:
    python3 scripts/ocr_name_benchmark.py --folder \"path/to/exam\"
    python3 scripts/ocr_name_benchmark.py --folder \"...\" --pdf path/to/cleaned_scan.pdf
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.console import Console
from rich.markup import escape
from rich.table import Table

# Repo root for imports when run as a script
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from extraction.ground_truth import fuzzy_match_name  # noqa: E402
from shared.exam_paths import find_latest_cleaned_scan  # noqa: E402
from shared.load_student_list import read_student_list  # noqa: E402

DEFAULT_DPI = 300
DEFAULT_STRIP = (0.0, 0.0, 840.0, 125.0)


def pts_to_px(pts: float, dpi: int) -> int:
    """Convert PDF points to pixels at *dpi* (1 pt = 1/72 inch)."""
    return round(pts * dpi / 72)


def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    return f"{seconds * 1000:.0f} ms"


def load_strip_pil(
    pdf_path: Path,
    dpi: int,
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Image.Image:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        pg = doc[page]
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = pg.get_pixmap(clip=clip, dpi=dpi)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def preprocess(pil_img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    import cv2

    gray_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return Image.fromarray(dilated), dilated


def run_tesseract(pil_img: Image.Image) -> str:
    try:
        import pytesseract
    except ImportError:
        return "[pytesseract not installed]"

    configs = ("--psm 11 --oem 3", "--psm 6 --oem 3")
    best = ""
    try:
        for cfg in configs:
            txt = pytesseract.image_to_string(pil_img, config=cfg).strip()
            if len(txt) > len(best):
                best = txt
    except Exception as exc:  # noqa: BLE001 — binary missing, bad image, etc.
        return f"[tesseract error: {exc}]"
    return best


def run_easyocr(cv2_img: np.ndarray) -> str:
    try:
        import easyocr
    except ImportError:
        return "[easyocr not installed]"

    if not hasattr(run_easyocr, "_reader"):
        run_easyocr._reader = easyocr.Reader(["en"], gpu=False, verbose=False)  # type: ignore[attr-defined]

    reader = run_easyocr._reader  # type: ignore[attr-defined]
    results = reader.readtext(cv2_img, detail=1)
    fragments = [text for (_, text, conf) in results if conf >= 0.3]
    return " ".join(fragments)


def run_paddleocr(cv2_img: np.ndarray) -> str:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "[paddleocr not installed]"

    if not hasattr(run_paddleocr, "_ocr"):
        try:
            run_paddleocr._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)  # type: ignore[attr-defined]
        except TypeError:
            run_paddleocr._ocr = PaddleOCR(use_angle_cls=True, lang="en")  # type: ignore[attr-defined]

    ocr = run_paddleocr._ocr  # type: ignore[attr-defined]
    try:
        result = ocr.ocr(cv2_img, cls=True)
    except Exception as exc:  # noqa: BLE001 — surface Paddle API/version issues
        return f"[paddleocr error: {exc}]"

    if not result or result == [None]:
        return ""

    fragments: list[str] = []
    for page_result in result:
        if page_result is None:
            continue
        for line in page_result:
            if len(line) < 2:
                continue
            _box, payload = line[0], line[1]
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                text, conf = payload[0], payload[1]
                if conf >= 0.3:
                    fragments.append(str(text))
    return " ".join(fragments)


def search_paths_hint(exam_folder: Path, output_base: Path) -> str:
    stem = exam_folder.name.replace(" ", "_")
    return (
        f"  {output_base / stem / 'cleaned_scan.pdf'}\n"
        f"  {output_base / stem}/*/cleaned_scan.pdf\n"
        f"  {exam_folder / 'cleaned_scan.pdf'}"
    )


def benchmark(
    pdf_path: Path,
    students: list[str],
    dpi: int,
    page: int,
    strip: tuple[float, float, float, float],
    console: Console,
    *,
    time_strip: bool = True,
) -> None:
    x0, y0, x1, y1 = strip

    if time_strip:
        t0 = time.perf_counter()
    pil_strip = load_strip_pil(pdf_path, dpi, page, x0, y0, x1, y1)
    pil_pre, cv2_pre = preprocess(pil_strip)
    if time_strip:
        dt = time.perf_counter() - t0
        console.print(f"[dim]Strip load + preprocess:[/dim] {format_duration(dt)}")

    console.print(
        f"[dim]Strip size:[/dim] {pil_pre.size[0]} × {pil_pre.size[1]} px  "
        f"([dim]PDF pts[/dim] {x0},{y0} → {x1},{y1} @ {dpi} DPI)\n"
    )

    engines: list[tuple[str, Any]] = [
        ("Tesseract", lambda: run_tesseract(pil_pre)),
        ("EasyOCR", lambda: run_easyocr(cv2_pre)),
        ("PaddleOCR", lambda: run_paddleocr(cv2_pre)),
    ]

    rows: list[tuple[str, str, str, str]] = []

    for name, fn in engines:
        t0 = time.perf_counter()
        raw = fn()
        elapsed = time.perf_counter() - t0
        raw_s = raw if isinstance(raw, str) else ""
        if raw_s.startswith("[") and (
            "not installed" in raw_s or "error:" in raw_s.lower()
        ):
            match_s = "—"
        else:
            matched = fuzzy_match_name(raw_s.strip(), students)
            match_s = matched if matched is not None else "no match"
        raw_display = raw_s if len(raw_s) <= 80 else raw_s[:77] + "..."
        # Rich treats "[" as markup; escape so "[easyocr not installed]" is visible.
        rows.append((name, escape(raw_display), match_s, format_duration(elapsed)))

    table = Table(title="OCR name benchmark", show_header=True, header_style="bold")
    table.add_column("Engine", style="cyan", no_wrap=True)
    table.add_column("Raw OCR", overflow="fold")
    table.add_column("Match")
    table.add_column("Time", justify="right")

    for r in rows:
        table.add_row(*r)

    console.print(table)
    console.print(
        "\n[dim]Times include lazy init on first EasyOCR/PaddleOCR use in this process. "
        "Fuzzy matching is not timed.[/dim]\n"
        "[dim]If EasyOCR or PaddleOCR are not installed in this Python, run: "
        "`source paddle_env/bin/activate` (see scripts/install_paddleocr.sh).[/dim]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark Tesseract / EasyOCR / PaddleOCR on a name strip vs StudentList.xlsx",
    )
    ap.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Exam folder containing StudentList.xlsx",
    )
    ap.add_argument("--pdf", type=Path, default=None, help="Override PDF (default: latest cleaned_scan.pdf)")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Render DPI (default {DEFAULT_DPI})")
    ap.add_argument("--page", type=int, default=0, help="PDF page index (0-based)")
    ap.add_argument("--output-base", type=Path, default=Path("output"), help="Output tree root (default output)")
    ap.add_argument("--strip-x0", type=float, default=DEFAULT_STRIP[0], help="Strip left (pt)")
    ap.add_argument("--strip-y0", type=float, default=DEFAULT_STRIP[1], help="Strip top (pt)")
    ap.add_argument("--strip-x1", type=float, default=DEFAULT_STRIP[2], help="Strip right (pt)")
    ap.add_argument("--strip-y1", type=float, default=DEFAULT_STRIP[3], help="Strip bottom (pt)")
    ap.add_argument("--no-strip-timing", action="store_true", help="Do not print load+preprocess time")
    args = ap.parse_args()

    console = Console()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        console.print(f"[red]ERROR:[/red] folder not found: {folder}")
        raise SystemExit(1)

    try:
        students = read_student_list(folder)
    except (FileNotFoundError, ImportError) as e:
        console.print(f"[red]ERROR:[/red] {e}")
        raise SystemExit(1)

    if not students:
        console.print("[red]ERROR:[/red] no names loaded from roster")
        raise SystemExit(1)

    pdf = args.pdf
    if pdf is None:
        found = find_latest_cleaned_scan(folder, output_base=args.output_base)
        if found is None:
            console.print(
                "[red]No cleaned_scan.pdf found.[/red] Searched:\n"
                + search_paths_hint(folder, args.output_base.expanduser().resolve())
            )
            console.print("\n[yellow]Tip:[/yellow] pass --pdf path/to/cleaned_scan.pdf or run from repo root.")
            raise SystemExit(1)
        pdf = found
    else:
        pdf = pdf.expanduser().resolve()

    if not pdf.is_file():
        console.print(f"[red]ERROR:[/red] PDF not found: {pdf}")
        raise SystemExit(1)

    strip = (args.strip_x0, args.strip_y0, args.strip_x1, args.strip_y1)

    console.print(f"[bold]Source[/bold] {pdf}")
    console.print(f"[bold]Roster[/bold] {len(students)} names from [dim]StudentList.xlsx[/dim]\n")

    benchmark(
        pdf,
        students,
        args.dpi,
        args.page,
        strip,
        console,
        time_strip=not args.no_strip_timing,
    )


if __name__ == "__main__":
    main()
