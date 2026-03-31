#!/usr/bin/env python3
"""
Repeated ``--first-students`` eval for iterative tuning (intended to run on ``dev``).

Runs ``extract_answers.py`` in a subprocess so each iteration exercises the real CLI.
Append-only JSONL log with accuracy per run. Stop after a wall-clock budget (e.g. 2–4 hours).

Typical workflow with an external agent: between iterations, edit prompts or helpers in
``extract_answers.py`` on ``dev``, then let this script re-measure. You can also vary
``EXTRACT_CROP_FRACTION`` or ``PDF_DPI`` in the environment between runs.

Example (4 hours, 12 students, 45 minutes between runs):

    export GOOGLE_API_KEY=...
    python benchmark_eval_loop.py output/exam.pdf --n 12 --max-hours 4 --sleep-seconds 2700
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _run_once(repo_root: Path, pdf: Path, n: int, python_exe: str) -> tuple[int, Path | None]:
    stem = pdf.stem
    eval_path = repo_root / f"{stem}_first{n}_eval.json"
    cmd = [python_exe, str(repo_root / "extract_answers.py"), str(pdf), "--first-students", str(n)]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
    )
    if eval_path.exists():
        return proc.returncode, eval_path
    return proc.returncode, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-bounded eval loop for extract_answers.py")
    parser.add_argument("pdf", type=Path, help="Input PDF (same as extract_answers.py)")
    parser.add_argument("--n", type=int, default=12, metavar="N", help="First N students (default: 12)")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=4.0,
        help="Stop after this many wall-clock hours (default: 4)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pause between successful iterations (default: 0)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("eval_benchmark_log.jsonl"),
        help="Append-only JSONL log path (default: eval_benchmark_log.jsonl)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    pdf = args.pdf if args.pdf.is_absolute() else repo_root / args.pdf
    if not pdf.exists():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        sys.exit(1)

    budget_s = max(0.0, args.max_hours * 3600.0)
    deadline = time.perf_counter() + budget_s
    iteration = 0
    log_path = args.log if args.log.is_absolute() else repo_root / args.log

    print(
        f"Eval loop: pdf={pdf} n={args.n} max_hours={args.max_hours} "
        f"sleep_s={args.sleep_seconds} log={log_path}",
        flush=True,
    )

    while time.perf_counter() < deadline:
        iteration += 1
        t0 = time.perf_counter()
        rc, eval_json = _run_once(repo_root, pdf, args.n, sys.executable)
        elapsed = time.perf_counter() - t0

        row: dict = {
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "returncode": rc,
            "elapsed_seconds": round(elapsed, 2),
            "pdf": str(pdf),
            "n": args.n,
            "eval_json": str(eval_json) if eval_json else None,
            "cumulative_accuracy_pct": None,
            "cumulative_correct": None,
            "cumulative_total": None,
        }

        if eval_json and eval_json.exists():
            try:
                with open(eval_json, encoding="utf-8") as f:
                    payload = json.load(f)
                summ = payload.get("summary") or {}
                row["cumulative_accuracy_pct"] = summ.get("cumulative_accuracy_pct")
                row["cumulative_correct"] = summ.get("cumulative_correct")
                row["cumulative_total"] = summ.get("cumulative_total")
            except (OSError, json.JSONDecodeError, TypeError) as e:
                row["log_error"] = str(e)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        acc = row.get("cumulative_accuracy_pct")
        print(
            f"[iter {iteration}] rc={rc} elapsed={elapsed:.1f}s "
            f"accuracy={acc}% correct={row.get('cumulative_correct')}/"
            f"{row.get('cumulative_total')}",
            flush=True,
        )

        if time.perf_counter() >= deadline:
            break
        sleep_left = deadline - time.perf_counter()
        if sleep_left <= 0:
            break
        to_sleep = min(args.sleep_seconds, sleep_left)
        if to_sleep > 0:
            print(f"Sleeping {to_sleep:.0f}s ...", flush=True)
            time.sleep(to_sleep)

    print("Eval loop finished (time budget exhausted).", flush=True)


if __name__ == "__main__":
    main()
