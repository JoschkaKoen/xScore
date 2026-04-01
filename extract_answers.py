#!/usr/bin/env python3
"""
extract_answers.py
------------------
CLI entry point: extracts student names + handwritten MC answers from scanned
exam PDFs using the configured vision model. See ``extraction/`` for modules.

    python extract_answers.py
    python extract_answers.py output/some_other.pdf
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pdf2image import convert_from_path

from config import (
    AI_MODEL,
    API_CALL_DELAY_S,
    DEFAULT_PDF,
    PDF_DPI,
    SAVE_DEBUG_IMAGES,
)

from extraction.eval import extract_first_n_students_eval
from extraction.ground_truth import (
    calculate_student_accuracy,
    fuzzy_match_name,
    load_ground_truth,
)
from extraction.images import (
    crop_top,
    effective_crop_fraction,
    preprocess_for_extraction,
    to_jpeg_bytes,
)
from extraction.profiles import get_profile
from extraction.providers import call_ocr_api, create_extraction_client
from extraction.reporting import (
    Colors,
    color_wrong_answer,
    format_accuracy,
    generate_report_pdf,
    load_existing_results,
    print_summary,
    save_results,
)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract handwritten answers from scanned exam PDFs (Gemini or Kimi per config)."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=DEFAULT_PDF,
        help=f"Path to input PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        default=False,
        help="Skip pages already present in the resume JSON (default: off, re-process everything)",
    )
    parser.add_argument(
        "--first-students",
        type=int,
        default=6,
        metavar="N",
        help="Only process the first N pages: extract, compare to ground truth, print summary, then exit. (default: 6)",
    )
    args = parser.parse_args()

    print(f"[INFO] Using AI Model: {AI_MODEL}")

    pdf_path = Path(args.pdf)
    profile = get_profile()
    answer_fields = profile.answer_fields

    if args.first_students > 0:
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            raise SystemExit(1)
        client = create_extraction_client()
        if client is None:
            if AI_MODEL.startswith("kimi"):
                print("ERROR: Kimi model selected but failed to initialize. Check KIMI_API_KEY.")
            else:
                print("ERROR: Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env or environment.")
            raise SystemExit(1)
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            print("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        stem = pdf_path.stem
        eval_json = Path(f"{stem}_first{args.first_students}_eval.json")
        result = extract_first_n_students_eval(
            pdf_path,
            n=args.first_students,
            client=client,
            save_results_path=eval_json,
            verbose=True,
        )
        print(
            f"\nEval summary: {result['summary']['cumulative_correct']}/"
            f"{result['summary']['cumulative_total']} correct "
            f"({result['summary']['cumulative_accuracy_pct']:.1f}%)"
        )
        raise SystemExit(0)

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        raise SystemExit(1)

    stem = pdf_path.stem
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{stem}_answers.json"
    output_tex = output_dir / f"{stem}_answers.tex"
    output_report = output_dir / f"{stem}_answers.pdf"
    debug_image_dir = Path(f"debug/debug_crops_{stem}")

    client = create_extraction_client()
    if client is None:
        if AI_MODEL.startswith("kimi"):
            print("ERROR: Kimi model selected but failed to initialize. Check KIMI_API_KEY.")
        else:
            print("ERROR: Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env or environment.")
        raise SystemExit(1)

    if SAVE_DEBUG_IMAGES:
        debug_image_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(output_json) if args.skip else {}
    if existing:
        print(f"Resuming -- {len(existing)} pages already done, skipping them.")
    elif args.skip:
        print("No existing results found, processing all pages.")

    ground_truth = load_ground_truth()
    gt_names = list(ground_truth.keys())
    if ground_truth:
        print(f"Ground truth loaded: {len(ground_truth)} students ({', '.join(gt_names)})")
    else:
        print("Warning: No ground truth found. Accuracy metrics disabled.")

    print(f"Converting PDF to images at {PDF_DPI} DPI (this may take a minute)...")
    t_pdf_to_images = time.perf_counter()
    pages = convert_from_path(str(pdf_path), dpi=PDF_DPI, thread_count=os.cpu_count())
    pdf_to_images_s = time.perf_counter() - t_pdf_to_images
    n_pages = len(pages)
    per_page = pdf_to_images_s / n_pages if n_pages else 0.0
    print(
        f"PDF→images ({PDF_DPI} DPI): {pdf_to_images_s:.2f}s total "
        f"({per_page:.2f}s/page) — {n_pages} pages.\n"
    )

    results_map: dict[int, dict] = dict(existing)

    cumulative_correct = 0
    cumulative_total = 0
    students_with_gt = 0

    for page_num, page in enumerate(pages, start=1):
        if page_num in results_map:
            print(f"  Page {page_num:3d}/{len(pages)} -- skipped (already processed)")
            continue

        print(f"  Page {page_num:3d}/{len(pages)} -- extracting...", end="", flush=True)

        crop = crop_top(page, effective_crop_fraction())
        processed = preprocess_for_extraction(crop)
        img_bytes = to_jpeg_bytes(processed)

        if SAVE_DEBUG_IMAGES:
            processed.save(debug_image_dir / f"page_{page_num:04d}.jpg", quality=85)

        data = call_ocr_api(client, img_bytes, page_num, profile)
        data["page_number"] = page_num
        results_map[page_num] = data

        conf = data.get("confidence", "?")
        name = data.get("student_name", "?")
        marker = {"high": "OK", "medium": "??", "low": "!!", "failed": "XX"}.get(conf, "??")
        q38lt_raw = data.get("q38_left_top", "?")
        q39l_raw = data.get("q39_left", "?")
        q40l_raw = data.get("q40_left", "?")
        q38lb_raw = data.get("q38_left_bottom", "?")
        q39r_raw = data.get("q39_right", "?")
        q40r_raw = data.get("q40_right", "?")

        nc = data.get("student_name_confidence", "?")[0].upper() if data.get("student_name_confidence") else "?"
        c38lt = data.get("q38_left_top_confidence", "?")[0].upper() if data.get("q38_left_top_confidence") else "?"
        c39l = data.get("q39_left_confidence", "?")[0].upper() if data.get("q39_left_confidence") else "?"
        c40l = data.get("q40_left_confidence", "?")[0].upper() if data.get("q40_left_confidence") else "?"
        c38lb = data.get("q38_left_bottom_confidence", "?")[0].upper() if data.get("q38_left_bottom_confidence") else "?"
        c39r = data.get("q39_right_confidence", "?")[0].upper() if data.get("q39_right_confidence") else "?"
        c40r = data.get("q40_right_confidence", "?")[0].upper() if data.get("q40_right_confidence") else "?"

        student_acc_str = "N/A"
        cumulative_acc_str = "N/A"
        q38lt, q39l, q40l, q38lb, q39r, q40r = q38lt_raw, q39l_raw, q40l_raw, q38lb_raw, q39r_raw, q40r_raw

        if ground_truth and name not in ("UNKNOWN", "EXTRACTION_ERROR", "?"):
            matched_gt_name = fuzzy_match_name(name, gt_names)
            if matched_gt_name:
                gt_answers = ground_truth[matched_gt_name]
                gt_pad = list(gt_answers) + [""] * len(answer_fields)
                gt_pad = gt_pad[: len(answer_fields)]

                q38lt = color_wrong_answer(q38lt_raw, gt_pad[0])
                q39l = color_wrong_answer(q39l_raw, gt_pad[1])
                q40l = color_wrong_answer(q40l_raw, gt_pad[2])
                q38lb = color_wrong_answer(q38lb_raw, gt_pad[3])
                q39r = color_wrong_answer(q39r_raw, gt_pad[4])
                q40r = color_wrong_answer(q40r_raw, gt_pad[5])

                student_acc = calculate_student_accuracy(data, gt_pad, answer_fields)
                student_acc_str = format_accuracy(student_acc)

                for i, field in enumerate(answer_fields):
                    extracted_val = data.get(field, "?").upper().strip()
                    gt_val = gt_answers[i].upper().strip() if i < len(gt_answers) else ""
                    cumulative_total += 1
                    if extracted_val == gt_val and extracted_val not in ("", "?"):
                        cumulative_correct += 1

                students_with_gt += 1
                cumulative_acc = (cumulative_correct / cumulative_total * 100) if cumulative_total > 0 else 0
                cumulative_acc_str = format_accuracy(cumulative_acc)

                data["student_accuracy"] = student_acc
                data["matched_ground_truth_name"] = matched_gt_name

        print(
            f" [{marker}] {name}({nc})  |  Q38L↑:{q38lt}({c38lt})  Q39L:{q39l}({c39l})  "
            f"Q40L:{q40l}({c40l})  Q38L↓:{q38lb}({c38lb})  Q39R:{q39r}({c39r})  Q40R:{q40r}({c40r})  "
            f"|  Acc here: {student_acc_str} / Cum: {cumulative_acc_str}{Colors.RESET}"
        )

        save_results(list(results_map.values()), output_json)
        time.sleep(API_CALL_DELAY_S)

    all_results = list(results_map.values())
    print_summary(all_results, ground_truth if ground_truth else None, answer_fields=answer_fields)
    generate_report_pdf(all_results, output_tex, output_report)
    print(f"\nJSON  -> {output_json}")
    print(f"LaTeX -> {output_tex}")
    print(f"PDF   -> {output_report}")
    if SAVE_DEBUG_IMAGES:
        print(f"Crops -> {debug_image_dir}/")


if __name__ == "__main__":
    main()
