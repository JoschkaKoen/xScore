"""TTY-aware ANSI colors and optional emoji for CLI output.

Respects `NO_COLOR` (https://no-color.org/) and `FORCE_COLOR`. Set `ASCII_LOG=1` to
disable emoji (ASCII fallbacks). Set ``PIPELINE_VERBOSE=1`` or ``GRADE_VERBOSE=1`` for
wide step banners and extra detail from some modules. Colors follow the real terminal
attached behind any ``_stdout`` wrapper (e.g. ``grade.py``'s log tee).
"""

from __future__ import annotations

import os
import sys

# ANSI SGR codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"

_EMOJI = {
    "ok": "✓",
    "warn": "⚠",
    "err": "✗",
    "info": "›",
    "step": "▸",
    "doc": "📄",
    "folder": "📁",
    "users": "👥",
    "gear": "⚙",
    "search": "🔍",
    "broom": "🧹",
    "chart": "📊",
    "flag": "🏁",
    "spark": "✨",
}

_ASCII_FALLBACK = {
    "ok": "[+]",
    "warn": "[!]",
    "err": "[x]",
    "info": ">",
    "step": ">",
    "doc": "[#]",
    "folder": "[/]",
    "users": "[@]",
    "gear": "[*]",
    "search": "[?]",
    "broom": "[~]",
    "chart": "[=]",
    "flag": "[.]",
    "spark": "*",
}


def _chain_to_real_stream(stream: object) -> object:
    seen: set[int] = set()
    while stream is not None and id(stream) not in seen:
        seen.add(id(stream))
        inner = getattr(stream, "_stdout", None)
        if inner is None or inner is stream:
            break
        stream = inner
    return stream


def use_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    real = _chain_to_real_stream(sys.stdout)
    try:
        return bool(getattr(real, "isatty", lambda: False)())
    except (OSError, ValueError):
        return False


def use_emoji() -> bool:
    if os.environ.get("ASCII_LOG", "").strip():
        return False
    if not use_color():
        return False
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return enc.lower().startswith("utf")


def icon(name: str) -> str:
    if use_emoji():
        return _EMOJI.get(name, "•")
    return _ASCII_FALLBACK.get(name, "*")


def paint(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI codes; no-op when colors disabled."""
    if not use_color() or not codes:
        return text
    return "".join(codes) + text + RESET


def rule(char: str = "═", width: int = 60) -> str:
    line = char * width
    return paint(line, DIM) if use_color() else line


def pipeline_verbose() -> bool:
    """True when ``PIPELINE_VERBOSE`` or ``GRADE_VERBOSE`` requests heavy banners / debug."""
    v = (os.environ.get("PIPELINE_VERBOSE") or os.environ.get("GRADE_VERBOSE") or "").strip()
    return v.lower() in ("1", "true", "yes", "on")


def pipeline_step(readme_step: int, title: str, *, width: int = 60) -> None:
    """Print a pipeline step header (compact by default; wide rules if :func:`pipeline_verbose`)."""
    print()
    label = f"  {icon('step')}  Step {readme_step} — {title}"
    if pipeline_verbose():
        print(rule("═", width))
        print(paint(label, CYAN, BOLD))
        print(rule("═", width))
    else:
        print(paint(label, CYAN, BOLD))


def progress_line(message: str) -> None:
    """Highlight the current action (brighter than :func:`info_line`)."""
    print(paint(f"  {icon('info')}  {message}", CYAN), flush=True)


def info_line(message: str, *, key: str = "info") -> None:
    print(paint(f"  {icon(key)}  {message}", DIM))


def ok_line(message: str) -> None:
    print(paint(f"  {icon('ok')}  {message}", GREEN), flush=True)


def warn_line(message: str) -> None:
    print(paint(f"  {icon('warn')}  {message}", YELLOW))


def err_line(message: str) -> None:
    print(paint(f"  {icon('err')}  {message}", RED), file=sys.stderr)


def note_line(message: str) -> None:
    """Neutral highlighted line (e.g. paths)."""
    print(paint(f"  {icon('doc')}  {message}", BLUE))


def tool_line(tool: str, message: str) -> None:
    """Sub-system tag aligned with other status lines, e.g. ``  [deskew] …``."""
    tag = f"[{tool}]"
    indent = "  "
    if use_color():
        print(f"{indent}{paint(tag, MAGENTA, BOLD)} {message}", flush=True)
    else:
        print(f"{indent}{tag} {message}", flush=True)
