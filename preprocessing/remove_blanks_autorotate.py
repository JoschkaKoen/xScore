"""Rotate scanned exam PDFs upright (Tesseract OSD) and drop blank pages (pikepdf).

Used by :mod:`preprocessing.start_scan` before fine deskew. Formerly ``autograder.py``.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pikepdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from rich import box
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Default OSD raster DPI when process_pdf is called without analysis_dpi (standalone / tests).
# Pipeline callers must pass pipeline dpi: low DPI yields orientation_conf < 2.0 and skipped rotation.
ANALYSIS_DPI = 150
BLANK_DPI = 72  # Fast blank-page pass (mean/std); OSD uses *analysis_dpi* in a second render.
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


def _rotation_worker(args: tuple[int, Image.Image]) -> tuple[int, int]:
    page_num, pil_img = args
    return page_num, detect_rotation(pil_img)


def _raster_with_spinner(label: str, fn, *, console) -> list:
    """Run *fn()* in a background thread, show a spinner, return the result."""
    from shared.terminal_ui import format_duration, ok_line

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn)
        with Progress(
            TextColumn("  "),
            SpinnerColumn(),
            TextColumn("  {task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            prog.add_task(label, total=None)
            while not future.done():
                time.sleep(0.05)
    result = future.result()
    ok_line(f"{label} · {format_duration(time.perf_counter() - t0)}")
    return result


def process_pdf(
    input_path: str,
    output_path: str,
    analysis_dpi: int = ANALYSIS_DPI,
    blank_mean: float = BLANK_MEAN_THRESHOLD,
    blank_std: float = BLANK_STD_THRESHOLD,
    *,
    verbose: bool = True,
) -> None:
    """Blank detection + OSD rotation; write lossless PDF to *output_path*.

    Two Poppler passes: all pages at :data:`BLANK_DPI` for blanks, then all pages at
    *analysis_dpi* (pipeline DPI from :mod:`preprocessing.start_scan`) for Tesseract OSD.
    Parallel workers only run Tesseract on content pages; no per-page Poppler launches.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    from shared.terminal_ui import (
        PROGRESS_TASK_TEXT,
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
                f"[bold cyan]{icon('doc')}  PDF prep[/]",
                border_style="dim cyan",
            )
        )
        note_line(
            f"Blank pass @ {BLANK_DPI} DPI  |  OSD pass @ {analysis_dpi} DPI  |  "
            f"Blank: mean≥{blank_mean}, std≤{blank_std}"
        )

    _tc = os.cpu_count() or 4
    if verbose:
        c.print(
            f"\n[bold cyan]  {icon('broom')}  Pass 1: blank detection @ {BLANK_DPI} DPI[/]"
        )
        low_res_pages = convert_from_path(
            str(input_path),
            dpi=BLANK_DPI,
            grayscale=True,
            thread_count=_tc,
        )
    else:
        low_res_pages = _raster_with_spinner(
            f"Blank detection ({BLANK_DPI} DPI)",
            lambda: convert_from_path(str(input_path), dpi=BLANK_DPI, grayscale=True, thread_count=_tc),
            console=c,
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

    if verbose:
        c.print(
            f"\n[bold cyan]  {icon('gear')}  Pass 2: raster @ {analysis_dpi} DPI for OSD[/]"
        )
        hi_res_pages = convert_from_path(
            str(input_path),
            dpi=analysis_dpi,
            grayscale=True,
            thread_count=_tc,
        )
    else:
        hi_res_pages = _raster_with_spinner(
            f"Rotation detection ({analysis_dpi} DPI)",
            lambda: convert_from_path(str(input_path), dpi=analysis_dpi, grayscale=True, thread_count=_tc),
            console=c,
        )

    num_workers = min(_tc, len(content_page_nums))
    if verbose:
        c.print(
            f"  Tesseract OSD on {len(content_page_nums)} content pages "
            f"({num_workers} workers)"
        )

    rotation_map: dict[int, int] = {}
    osd_inputs = [(pn, hi_res_pages[pn - 1]) for pn in content_page_nums]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_rotation_worker, item): item[0] for item in osd_inputs
        }
        with Progress(
            TextColumn(PROGRESS_TASK_TEXT),
            BarColumn(bar_width=28),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=c,
            transient=False,
        ) as prog:
            task_id = prog.add_task("", total=len(futures))
            for future in as_completed(futures):
                page_num, angle = future.result()
                rotation_map[page_num] = angle
                prog.advance(task_id)

    del hi_res_pages

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
        rotated = sum(1 for a in rotation_map.values() if a != 0)
        rot_s = f"{rotated} rotated" if rotated else "none rotated"

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
        note_line("Saving cleaned PDF …")

    out_pdf.save(str(output_path))
    out_pdf.close()
    src_pdf.close()

    if verbose:
        ok_line("Blank drop + rotation complete.")
        c.print()
    else:
        kept = len(content_page_nums)
        blanks = len(blank_page_nums)
        page_s = f"{kept} of {total_pages}" if blanks else str(kept)
        ok_line(f"{page_s} pages  ·  {rot_s}")
