"""CLI output via Rich (tables/panels use :func:`get_console`).

Respects ``NO_COLOR`` and ``FORCE_COLOR``. Set ``ASCII_LOG=1`` to disable emoji.
"""

from __future__ import annotations

import logging
import os
import sys

from rich.console import Console
from rich.progress import ProgressColumn, Task
from rich.text import Text

# Legacy ANSI constants (some callers still pass these to :func:`paint`)
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


# Rich ``TextColumn`` template: aligns task labels with :func:`info_line` / :func:`tool_line`
# (they print ``  {icon}  {text}`` — two spaces + one-column icon + two spaces).
# Four spaces: Rich adds one cell padding before the first column, so five visible columns match.
PROGRESS_TASK_TEXT = "    {task.description}"


def format_duration(seconds: float) -> str:
    """Short duration for CLI (e.g. ``3.1s``, ``1m 5s``)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


class CompactElapsedColumn(ProgressColumn):
    """Elapsed wall time without Rich's ``0:00:`` zero-filled prefix (uses :func:`format_duration`)."""

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished_time is not None else task.elapsed
        if elapsed is None:
            elapsed = 0.0
        return Text(format_duration(float(elapsed)), style="progress.elapsed")


def get_console() -> Console:
    """Console writing to current ``sys.stdout`` (works with ``xscore.py`` tee)."""
    uc = use_color()
    return Console(
        file=sys.stdout,
        force_terminal=uc,
        no_color=not uc,
        emoji=use_emoji(),
        highlight=False,
        soft_wrap=True,
    )


def get_stderr_console() -> Console:
    uc = use_color()
    return Console(
        file=sys.stderr,
        force_terminal=uc,
        no_color=not uc,
        emoji=use_emoji(),
        highlight=False,
        soft_wrap=True,
    )


def pipeline_debug_ai() -> bool:
    """True when ``PIPELINE_DEBUG_AI`` requests stderr logging of truncated model responses."""
    v = (os.environ.get("PIPELINE_DEBUG_AI") or "").strip()
    return v.lower() in ("1", "true", "yes", "on")


_AI_LOG = logging.getLogger("autograder.ai")
_ai_log_handler_installed = False


def log_ai_response_debug(tag: str, model: str, raw: str) -> None:
    """Log first 500 chars of *raw* at DEBUG when ``PIPELINE_DEBUG_AI`` is set."""
    global _ai_log_handler_installed
    if not pipeline_debug_ai():
        return
    if not _ai_log_handler_installed:
        _ai_log_handler_installed = True
        _AI_LOG.setLevel(logging.DEBUG)
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        _AI_LOG.addHandler(h)
        _AI_LOG.propagate = False
    _AI_LOG.debug("%s model=%s raw_truncated=%r", tag, model, (raw or "")[:500])


def pipeline_step(
    readme_step: int,
    title: str,
    *,
    subtitle: str | None = None,
) -> None:
    """Print a compact pipeline step header."""
    c = get_console()
    label = f"  {icon('step')}  Step {readme_step} — {title}"
    c.print()
    c.print(f"[bold cyan]{label}[/]")
    if subtitle:
        c.print(f"[dim]  {icon('info')}  {subtitle}[/]")
    sys.stdout.flush()


def progress_line(message: str) -> None:
    """Highlight the current action (brighter than :func:`info_line`)."""
    c = get_console()
    c.print(f"[cyan]  {icon('info')}  {message}[/]")
    sys.stdout.flush()


def info_line(message: str, *, key: str = "info") -> None:
    get_console().print(f"[dim]  {icon(key)}  {message}[/]")


def ok_line(message: str) -> None:
    get_console().print(f"[green]  {icon('ok')}  {message}[/]")
    sys.stdout.flush()


def warn_line(message: str) -> None:
    get_console().print(f"[yellow]  {icon('warn')}  {message}[/]")


def err_line(message: str) -> None:
    get_stderr_console().print(f"[red]  {icon('err')}  {message}[/]")


def note_line(message: str) -> None:
    """Neutral highlighted line (e.g. paths)."""
    get_console().print(f"[blue]  {icon('doc')}  {message}[/]")


def tool_line(tool: str, message: str) -> None:
    """Like :func:`info_line` (tool name is accepted for call-site clarity only)."""
    _ = tool
    c = get_console()
    c.print(f"[dim]  {icon('info')}  {message}[/]")
    sys.stdout.flush()
