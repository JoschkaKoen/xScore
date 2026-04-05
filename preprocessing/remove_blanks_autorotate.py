"""Blank-page removal and page rotation for class-scan PDFs (pikepdf).

By default rotation follows each page's PDF ``/Rotate`` metadata (scanner output).
If the blank pass (72 DPI) renders a **content** page wider than tall, Poppler has
already applied a non-zero ``/Rotate`` to a portrait ``MediaBox``; we then write
``/Rotate=0`` on output so deskew sees portrait rasters. True landscape scans
with the same shape are rare here; use ``SCAN_USE_TESSERACT_ROTATION`` if needed.

Optional Tesseract OSD can add an extra CCW adjustment on top (slow); see
``config.SCAN_USE_TESSERACT_ROTATION``.

Used by :mod:`preprocessing.start_scan` before fine deskew. Formerly ``autograder.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pikepdf
from pdf2image import convert_from_path
from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from shared.terminal_ui import CompactElapsedColumn

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Default OSD raster DPI when process_pdf is called without analysis_dpi (standalone / tests).
# Pipeline callers should pass pipeline dpi when Tesseract rotation is enabled: low DPI yields
# orientation_conf < 2.0 and skipped rotation.
ANALYSIS_DPI = 150
BLANK_DPI = 72  # Fast blank-page pass (mean/std); optional second pass at *analysis_dpi* for OSD.
BLANK_MEAN_THRESHOLD = 250  # Pages with grayscale mean above this are considered blank (0-255)
BLANK_STD_THRESHOLD = 6     # Pages with grayscale std below this are considered blank


def _normalized_page_rotate(page: pikepdf.Page) -> int:
    try:
        r = int(page.get("/Rotate", 0))
    except (TypeError, ValueError):
        r = 0
    return r % 360


def _detect_rotation_osd(image: Image.Image) -> int:
    """Tesseract OSD: CCW rotation (0, 90, 180, 270) to upright the *raster*, or 0 on failure."""
    import pytesseract

    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))

        if confidence < 2.0:
            return 0

        return angle

    except pytesseract.TesseractError:
        return 0


# Public alias (legacy name; Tesseract path only).
detect_rotation = _detect_rotation_osd


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
    return page_num, _detect_rotation_osd(pil_img)


def _rotation_map_from_tesseract_osd(
    hi_res_pages: list,
    content_page_nums: list[int],
    *,
    console,
) -> dict[int, int]:
    """Run parallel Tesseract OSD on content pages; return page_num → extra CCW rotation."""
    from shared.terminal_ui import PROGRESS_TASK_TEXT

    _tc = os.cpu_count() or 4
    num_workers = min(_tc, len(content_page_nums))

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
            CompactElapsedColumn(),
            console=console,
            transient=False,
        ) as prog:
            task_id = prog.add_task("", total=len(futures))
            for future in as_completed(futures):
                page_num, angle = future.result()
                rotation_map[page_num] = angle
                prog.advance(task_id)

    return rotation_map


def _raster_with_spinner(label: str, fn, *, console) -> list:
    """Run *fn()* in a background thread, show a spinner, return the result."""
    from shared.terminal_ui import format_duration, info_line

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn)
        with Progress(
            TextColumn(" "),
            SpinnerColumn(),
            TextColumn(" {task.description}"),
            CompactElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            prog.add_task(label, total=None)
            while not future.done():
                time.sleep(0.05)
    result = future.result()
    info_line(f"{label} · {format_duration(time.perf_counter() - t0)}")
    return result


def detect_blank_page_lists(
    input_path: Path | str,
    *,
    blank_mean: float = BLANK_MEAN_THRESHOLD,
    blank_std: float = BLANK_STD_THRESHOLD,
) -> tuple[int, list[int], list[int], list[tuple[int, int]]]:
    """Raster at :data:`BLANK_DPI`, classify pages; return counts and render sizes per page."""
    input_path = Path(input_path)
    from shared.terminal_ui import get_console

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    c = get_console()
    _tc = os.cpu_count() or 4
    low_res_pages = _raster_with_spinner(
        f"Blank detection ({BLANK_DPI} DPI)",
        lambda: convert_from_path(
            str(input_path), dpi=BLANK_DPI, grayscale=True, thread_count=_tc
        ),
        console=c,
    )
    total_pages = len(low_res_pages)
    content_page_nums: list[int] = []
    blank_page_nums: list[int] = []
    for i, page_img in enumerate(low_res_pages):
        page_num = i + 1
        if is_blank_page(page_img, blank_mean, blank_std):
            blank_page_nums.append(page_num)
        else:
            content_page_nums.append(page_num)
    page_render_sizes: list[tuple[int, int]] = [img.size for img in low_res_pages]
    del low_res_pages
    return total_pages, content_page_nums, blank_page_nums, page_render_sizes


def write_rotated_pdf_after_blanks(
    input_path: Path | str,
    output_path: Path | str,
    *,
    total_pages: int,
    content_page_nums: list[int],
    blank_page_nums: list[int],
    page_render_sizes: list[tuple[int, int]],
    analysis_dpi: int = ANALYSIS_DPI,
    use_tesseract_rotation: bool | None = None,
) -> None:
    """Build PDF with blank pages dropped and rotation applied (scanner /Rotate or OSD)."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if use_tesseract_rotation is None:
        from config import SCAN_USE_TESSERACT_ROTATION

        use_tesseract_rotation = SCAN_USE_TESSERACT_ROTATION

    from shared.terminal_ui import err_line, get_console, ok_line, warn_line

    if input_path.resolve() == output_path.resolve():
        err_line(
            "Input and output paths are the same — refusing to overwrite the source PDF. "
            "Choose a different output path."
        )
        sys.exit(1)

    if not content_page_nums:
        warn_line("All pages were removed (all blank?). Nothing to save.")
        sys.exit(1)

    c = get_console()
    _tc = os.cpu_count() or 4
    src_pdf = pikepdf.open(str(input_path))
    landscape_pages: set[int] = set()

    if use_tesseract_rotation:
        hi_res_pages = _raster_with_spinner(
            f"Rotation detection ({analysis_dpi} DPI)",
            lambda: convert_from_path(
                str(input_path),
                dpi=analysis_dpi,
                grayscale=True,
                thread_count=_tc,
            ),
            console=c,
        )

        rotation_map = _rotation_map_from_tesseract_osd(
            hi_res_pages, content_page_nums, console=c
        )
        del hi_res_pages
    else:
        rotation_map = {pn: 0 for pn in content_page_nums}
        landscape_pages = {
            pn
            for pn in content_page_nums
            if page_render_sizes[pn - 1][0] > page_render_sizes[pn - 1][1]
        }

    if use_tesseract_rotation:
        rotated = sum(1 for a in rotation_map.values() if a != 0)
        rot_s = (
            f"{rotated} page(s) rotated via Tesseract OSD"
            if rotated
            else "All pages already upright"
        )
    else:
        rots: list[int] = []
        for pn in content_page_nums:
            if pn in landscape_pages:
                rots.append(0)
            else:
                rots.append(_normalized_page_rotate(src_pdf.pages[pn - 1]))
        cnt = Counter(rots)
        n_land = len(landscape_pages)
        n_kept = len(content_page_nums)
        if n_land == n_kept:
            rot_s = f"All {n_kept} pages corrected from landscape to portrait"
        elif n_land:
            rot_s = f"{n_land} of {n_kept} pages corrected from landscape to portrait"
        elif len(cnt) == 1:
            only_deg = next(iter(cnt))
            rot_s = (
                "All pages already upright"
                if only_deg == 0
                else f"All pages at {only_deg}°"
            )
        else:
            parts = [f"{n} at {deg}°" for deg, n in sorted(cnt.items())]
            rot_s = "Mixed orientation: " + ", ".join(parts)

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
        elif pn in landscape_pages:
            src_page["/Rotate"] = pikepdf.Integer(0)

        out_pdf.pages.append(src_page)

    out_pdf.save(str(output_path))
    out_pdf.close()
    src_pdf.close()

    kept = len(content_page_nums)
    blanks = len(blank_page_nums)
    if blanks:
        page_s = f"{kept} of {total_pages} pages  ·  {blanks} blank pages dropped"
    else:
        page_s = f"{kept} pages  ·  no blanks"
    ok_line(page_s)
    ok_line(rot_s)


def scan_blanks_state_to_json(
    *,
    source_pdf: Path,
    total_pages: int,
    content_page_nums: list[int],
    blank_page_nums: list[int],
    page_render_sizes: list[tuple[int, int]],
    blank_mean: float,
    blank_std: float,
    use_tesseract_rotation: bool,
    analysis_dpi: int,
) -> str:
    """Serialize blank-detection state for phased scan pipeline (step 5 → step 6)."""
    data = {
        "schema_version": 1,
        "source_pdf": str(source_pdf.resolve()),
        "total_pages": total_pages,
        "content_page_nums": content_page_nums,
        "blank_page_nums": blank_page_nums,
        "page_render_sizes": [list(s) for s in page_render_sizes],
        "blank_mean": blank_mean,
        "blank_std": blank_std,
        "use_tesseract_rotation": use_tesseract_rotation,
        "analysis_dpi": analysis_dpi,
    }
    return json.dumps(data, indent=2)


def scan_blanks_state_from_json(text: str) -> dict:
    data = json.loads(text)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported scan_blanks.json schema_version")
    sizes_raw = data["page_render_sizes"]
    page_render_sizes = [tuple(int(x) for x in pair) for pair in sizes_raw]
    data["page_render_sizes"] = page_render_sizes
    return data


def process_pdf(
    input_path: str,
    output_path: str,
    analysis_dpi: int = ANALYSIS_DPI,
    blank_mean: float = BLANK_MEAN_THRESHOLD,
    blank_std: float = BLANK_STD_THRESHOLD,
    *,
    use_tesseract_rotation: bool | None = None,
) -> None:
    """Blank detection; optional Tesseract OSD rotation; write PDF to *output_path*.

    Default (``use_tesseract_rotation`` false / ``config.SCAN_USE_TESSERACT_ROTATION`` off):
    one Poppler raster at :data:`BLANK_DPI` for blanks. Content pages whose render is
    landscape (width > height) get ``/Rotate`` cleared to 0 so downstream deskew gets
    portrait bitmaps; other pages keep scanner ``/Rotate``. No second raster;
    ``analysis_dpi`` unused.

    When ``use_tesseract_rotation`` is true: second raster at *analysis_dpi*, parallel
    Tesseract OSD, then ``existing /Rotate + OSD angle`` on each kept page (no landscape
    shortcut).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if use_tesseract_rotation is None:
        from config import SCAN_USE_TESSERACT_ROTATION

        use_tesseract_rotation = SCAN_USE_TESSERACT_ROTATION

    from shared.terminal_ui import err_line

    if input_path.resolve() == output_path.resolve():
        err_line(
            "Input and output paths are the same — refusing to overwrite the source PDF. "
            "Choose a different output path."
        )
        sys.exit(1)

    if not input_path.exists():
        err_line(f"Input file not found: {input_path}")
        sys.exit(1)

    total_pages, content_page_nums, blank_page_nums, page_render_sizes = (
        detect_blank_page_lists(
            input_path,
            blank_mean=blank_mean,
            blank_std=blank_std,
        )
    )

    write_rotated_pdf_after_blanks(
        input_path,
        output_path,
        total_pages=total_pages,
        content_page_nums=content_page_nums,
        blank_page_nums=blank_page_nums,
        page_render_sizes=page_render_sizes,
        analysis_dpi=analysis_dpi,
        use_tesseract_rotation=use_tesseract_rotation,
    )
