"""Resolve the exam folder from a CLI override, instruction hint, or heuristic search."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path


def find_folder(
    instruction_hint: str | None = None,
    cli_override: str | None = None,
    search_root: Path | None = None,
) -> Path:
    """Return the exam folder path.

    Priority:
    1. ``cli_override`` (--folder flag) — used as-is, must exist.
    2. ``instruction_hint`` — exact directory name match in ``search_root``.
    3. ``instruction_hint`` — fuzzy match against sub-directory names (≥ 0.6 ratio).
    4. Heuristic fallback: newest directory containing "test" or "exam" (case-insensitive)
       in ``search_root``.

    Raises ``FileNotFoundError`` if nothing is found.
    """
    root = search_root or Path.cwd()

    # 1. Explicit CLI override
    if cli_override:
        p = Path(cli_override)
        if not p.is_absolute():
            p = root / p
        if p.is_dir():
            return p.resolve()
        raise FileNotFoundError(f"--folder path does not exist or is not a directory: {p}")

    candidates = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # 2. Exact name match on hint
    if instruction_hint:
        hint_lower = instruction_hint.strip().lower()
        for d in candidates:
            if d.name.lower() == hint_lower:
                return d.resolve()

    # 3. Fuzzy match on hint
    if instruction_hint:
        hint_lower = instruction_hint.strip().lower()
        best: tuple[float, Path | None] = (0.0, None)
        for d in candidates:
            ratio = SequenceMatcher(None, hint_lower, d.name.lower()).ratio()
            # Also accept substring containment as a strong match
            if hint_lower in d.name.lower() or d.name.lower() in hint_lower:
                ratio = max(ratio, 0.75)
            if ratio > best[0]:
                best = (ratio, d)
        if best[0] >= 0.6 and best[1] is not None:
            return best[1].resolve()

    # 4. Heuristic: newest dir whose name contains "test" or "exam"
    exam_dirs = sorted(
        [d for d in candidates if any(kw in d.name.lower() for kw in ("test", "exam"))],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if exam_dirs:
        return exam_dirs[0].resolve()

    raise FileNotFoundError(
        f"Could not locate an exam folder in {root}. "
        "Use --folder to specify it explicitly."
    )
