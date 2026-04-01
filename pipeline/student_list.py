"""Read the student roster from an Excel file (.xlsx) in the exam folder."""

from __future__ import annotations

from pathlib import Path


def read_student_list(folder: Path) -> list[str]:
    """Return a list of student names from the first .xlsx found in *folder*.

    Search preference:
    1. Any .xlsx whose name contains "student" (case-insensitive).
    2. The first .xlsx file found otherwise.

    Names are read from the first column that contains string data,
    skipping blank cells and obvious header rows ("Name", "Student", etc.).

    Raises ``FileNotFoundError`` if no .xlsx file is present.
    Raises ``ImportError`` if openpyxl is not installed.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("openpyxl is required: pip install openpyxl>=3.1.0") from exc

    xlsx_files = list(folder.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx file found in {folder}")

    preferred = [f for f in xlsx_files if "student" in f.name.lower()]
    target = preferred[0] if preferred else xlsx_files[0]

    wb = openpyxl.load_workbook(target, read_only=True, data_only=True)
    ws = wb.active

    header_keywords = {"name", "student", "students", "names", "no", "number", "#"}
    names: list[str] = []

    for row in ws.iter_rows(values_only=True):
        # Find first cell with a string value
        for cell in row:
            if cell is None:
                continue
            val = str(cell).strip()
            if not val:
                continue
            # Skip header rows
            if val.lower() in header_keywords:
                break
            names.append(val)
            break  # only take first non-empty string cell per row

    wb.close()
    return names
