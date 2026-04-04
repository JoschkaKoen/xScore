"""Rotate scanned exam PDFs upright (Tesseract OSD) and drop blank pages (pikepdf).

Used by :mod:`preprocessing.start_scan` before fine deskew. Formerly ``autograder.py``.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pikepdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from rich import box
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

ANALYSIS_DPI = 150          # DPI for OSD only (Tesseract needs ~70+; 150 is plenty for 90° detection).
BLANK_DPI = 72              # Low DPI for fast blank-page detection.
BLANK_MEAN_THRESHOLD = 250  # Pages with grayscale mean above this are considered blank (0-255)
BLANK_STD_THRESHOLD = 6     # Pages with grayscale std below this are considered blank


def detect_rotation(image: Image.Image) -> int:
    """Return CCW rotation (0, 90, 180, 270) needed for upright page, or 0 on failure."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))

        if confidence < 2.0:
            return 0

        return angle

    except pytesseract.TesseractError:
        return 0


def is_blank_page(
    image: Image.Image,
    mean_threshold: float = BLANK_MEAN_THRESHOLD,
    std_threshold: float = BLANK_STD_THRESHOLD,
) -> bool:
    gray_img = image if image.mode == "L" else image.convert("L")
    gray = np.array(gray_img, dtype=np.float32)
    mean = gray.mean()
    std = gray.std()
    return (mean >= mean_threshold) and (std <= std_threshold)


def _osd_worker(page_num: int, input_path: str, dpi: int) -> tuple[int, int]:
    images = convert_from_path(
        input_path, dpi=dpi, first_page=page_num, last_page=page_num
    )
    angle = detect_rotation(images[0])
    return (page_num, angle)


def process_pdf(
    input_path: str,
    output_path: str,
    analysis_dpi: int = ANALYSIS_DPI,
    blank_mean: float = BLANK_MEAN_THRESHOLD,
    blank_std: float = BLANK_STD_THRESHOLD,
    *,
    verbose: bool = True,
) -> None:
    """Blank detection + OSD rotation; write lossless PDF to *output_path*."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    from shared.terminal_ui import (
        err_line,
        get_console,
        icon,
        note_line,
        ok_line,
        warn_line,
    )

    if input_path.resolve() == output_path.resolve():
        err_line(
            "Input and output paths are the same — refusing to overwrite the source PDF. "
            "Choose a different output path."
        )
        sys.exit(1)

    if not input_path.exists():
        err_line(f"Input file not found: {input_path}")
        sys.exit(1)

    c = get_console()

    if verbose:
        c.print()
        c.print(
            Panel(
                f"[bold cyan]{icon('doc')}  PDF prep  —  {input_path.name}[/]",
                border_style="dim cyan",
            )
        )
        note_line(f"Full path: {input_path}")
        note_line(
            f"Analysis DPI: {analysis_dpi}  |  Blank: mean≥{blank_mean}, std≤{blank_std}"
        )

    if verbose:
        c.print(
            f"\n[bold cyan]  {icon('broom')}  Pass 1: blank detection @ {BLANK_DPI} DPI[/]"
        )
    low_res_pages = convert_from_path(
        str(input_path), dpi=BLANK_DPI, grayscale=True, thread_count=os.cpu_count() or 4
    )
    total_pages = len(low_res_pages)
    if verbose:
        c.print(f"  Total pages: {total_pages}")

    content_page_nums: list[int] = []
    blank_page_nums: list[int] = []

    for i, page_img in enumerate(low_res_pages):
        page_num = i + 1
        if is_blank_page(page_img, blank_mean, blank_std):
            blank_page_nums.append(page_num)
        else:
            content_page_nums.append(page_num)

    if verbose:
        c.print(
            f"  → {len(blank_page_nums)} blank pages, {len(content_page_nums)} content pages"
        )

    del low_res_pages

    if not content_page_nums:
        warn_line("All pages were removed (all blank?). Nothing to save.")
        sys.exit(1)

    num_workers = min(os.cpu_count() or 4, len(content_page_nums))
    if verbose:
        c.print(
            f"\n[bold cyan]  {icon('gear')}  Pass 2: OSD rotation on {len(content_page_nums)} pages "
            f"@ {analysis_dpi} DPI ({num_workers} workers)[/]"
        )

    rotation_map: dict[int, int] = {}
    input_str = str(input_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_osd_worker, pn, input_str, analysis_dpi): pn
            for pn in content_page_nums
        }
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=c,
            transient=True,
        ) as prog:
            task_id = prog.add_task("OSD rotation", total=len(futures))
            for future in as_completed(futures):
                page_num, angle = future.result()
                rotation_map[page_num] = angle
                prog.advance(task_id)

    if verbose:
        rot_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            title=f"{icon('chart')}  Rotation summary",
            title_style="bold cyan",
        )
        rot_table.add_column("Page", justify="right", style="dim")
        rot_table.add_column("Status", overflow="fold")

        for pn in sorted(rotation_map):
            angle = rotation_map[pn]
            status = f"rotate {angle}°" if angle != 0 else "ok"
            rot_table.add_row(str(pn), status)
        for pn in sorted(blank_page_nums):
            rot_table.add_row(str(pn), "blank (removed)")

        c.print()
        c.print(Panel(rot_table, border_style="dim cyan"))

    else:
        angle_counts = Counter(rotation_map.values())
        parts: list[str] = []
        for angle in sorted(angle_counts):
            n = angle_counts[angle]
            if angle == 0:
                parts.append(f"{n} already upright")
            else:
                parts.append(f"{n} rotated")
        rot_s = ", ".join(parts) if parts else "—"

    src_pdf = pikepdf.open(str(input_path))
    out_pdf = pikepdf.new()

    for pn in content_page_nums:
        src_page = src_pdf.pages[pn - 1]
        angle = rotation_map.get(pn, 0)

        if angle != 0:
            try:
                existing_rotate = int(src_page.get("/Rotate", 0))
            except (TypeError, ValueError):
                existing_rotate = 0
            new_rotate = (existing_rotate + angle) % 360
            src_page["/Rotate"] = new_rotate

        out_pdf.pages.append(src_page)

    if verbose:
        note_line(f"Pages retained: {len(content_page_nums)}/{total_pages}")
        note_line(f"Writing: {output_path}")

    out_pdf.save(str(output_path))
    out_pdf.close()
    src_pdf.close()

    if verbose:
        ok_line("Passes 1–2 complete (rotate + de-blank).")
        c.print()
    else:
        _blank_s = f"{len(blank_page_nums)} blank" if blank_page_nums else "none blank"
        ok_line(
            f"{len(content_page_nums)}/{total_pages} pages kept ({_blank_s} removed)  ·  {rot_s}"
        )
