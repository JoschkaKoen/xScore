#!/usr/bin/env python3
"""
improvement_agent.py
--------------------
Continuously improves ``config.py`` and ``extraction/profiles/igcse_physics.py`` to maximize accuracy.
Runs for 2-4 hours, targeting 100% accuracy on first 6 students, then students 6-12.

Usage:
    source .venv/bin/activate
    python scripts/improvement_agent.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DEFAULT_PDF

# Configuration
IMPROVEMENT_LOG = _REPO_ROOT / "improvement_log.jsonl"
CONFIG_PATH = _REPO_ROOT / "config.py"
PROFILE_PATH = _REPO_ROOT / "extraction/profiles/igcse_physics.py"
TARGET_ACCURACY = 100.0
PHASE_1_START = 1  # First student
PHASE_1_END = 6    # Last student of phase 1
PHASE_2_START = 6  # First student of phase 2  
PHASE_2_END = 12   # Last student of phase 2

# Track best results
best_accuracy = 0.0
best_config = None
iteration = 0


def log_event(event_type: str, data: dict):
    """Log improvement events to file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "iteration": iteration,
        **data
    }
    with open(IMPROVEMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_eval(start_student: int = 1, end_student: int = 6) -> dict:
    """Run evaluation on specified student range and return results."""
    pdf_path = Path(DEFAULT_PDF)
    if not pdf_path.is_absolute():
        pdf_path = _REPO_ROOT / pdf_path

    # Use the --first-students flag for end_student
    # We'll filter for start_student later
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "extract_answers.py"),
        str(pdf_path),
        "--first-students",
        str(end_student),
    ]

    print(f"\n{'='*60}")
    print(f"Running eval: students {start_student}-{end_student}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse the result from the saved JSON
    eval_json = _REPO_ROOT / f"{pdf_path.stem}_first{end_student}_eval.json"
    if eval_json.exists():
        with open(eval_json) as f:
            data = json.load(f)
        
        # Filter for our student range if needed
        if start_student > 1:
            # Filter students to only include our range
            filtered_students = [
                s for s in data.get("students", [])
                if start_student <= s.get("page_number", 0) <= end_student
            ]
            # Recalculate accuracy
            total = sum(len(s.get("per_field", [])) for s in filtered_students)
            correct = sum(
                sum(1 for f in s.get("per_field", []) if f.get("correct"))
                for s in filtered_students
            )
            accuracy = (correct / total * 100) if total > 0 else 0.0
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "students": filtered_students,
                "summary": data.get("summary", {})
            }
        else:
            summary = data.get("summary", {})
            return {
                "accuracy": summary.get("cumulative_accuracy_pct", 0.0),
                "correct": summary.get("cumulative_correct", 0),
                "total": summary.get("cumulative_total", 0),
                "students": data.get("students", []),
                "summary": summary
            }
    
    return {"accuracy": 0.0, "correct": 0, "total": 0, "students": []}


def analyze_errors(result: dict) -> list:
    """Analyze which fields are being misclassified."""
    errors = []
    for student in result.get("students", []):
        name = student.get("matched_ground_truth_name", "UNKNOWN")
        for field in student.get("per_field", []):
            if not field.get("correct"):
                errors.append({
                    "student": name,
                    "page": student.get("page_number"),
                    "field": field.get("field"),
                    "extracted": field.get("extracted"),
                    "ground_truth": field.get("ground_truth")
                })
    return errors


def improve_prompt_v1():
    """Improvement: Enhance prompt with clearer field positioning guidance."""
    print("Applying improvement: Enhanced prompt with clearer field positioning...")
    
    # Read current file
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find and enhance the PROMPT section about layout
    old_layout = """LEFT COLUMN (middle-left area, vertical arrangement):
  Position 1 (upper): Question 38  → field: q38_left_top
  Position 2:         Question 39  → field: q39_left
  Position 3:         Question 40  → field: q40_left
  Position 4 (lower): Question 38  → field: q38_left_bottom (second instance of Q38)

RIGHT COLUMN (middle-right area, vertical arrangement):
  Position 1 (upper): Question 39  → field: q39_right
  Position 2 (lower): Question 40  → field: q40_right"""
    
    new_layout = """LEFT COLUMN (middle-left area, approximately 20-45% from left edge, 20-80% from top):
  Position 1 (upper third):     Question 38  → field: q38_left_top    
  Position 2 (middle):          Question 39  → field: q39_left
  Position 3 (lower middle):    Question 40  → field: q40_left
  Position 4 (bottom):          Question 38  → field: q38_left_bottom (SECOND instance of Q38)
  
  IMPORTANT: There are TWO Question 38s on the left side - one at top, one at bottom!

RIGHT COLUMN (middle-right area, approximately 55-80% from left edge, 20-80% from top):
  Position 1 (upper half):      Question 39  → field: q39_right
  Position 2 (lower half):      Question 40  → field: q40_right
  
  IMPORTANT: The right column only has Q39 and Q40, NOT Q38!"""
    
    if old_layout in content:
        content = content.replace(old_layout, new_layout)
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Applied enhanced layout guidance")
        return True
    print("  ✗ Could not find layout section to enhance")
    return False


def improve_prompt_v2():
    """Improvement: Add more examples of ambiguous cases."""
    print("Applying improvement: Add more ambiguous case examples...")
    
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add more examples after existing ones
    old_examples = """Example 5 - Name extraction:
  [Image shows "john smith" written at top]
  → student_name: "John Smith", student_name_confidence: "high"
"""
    
    new_examples = """Example 5 - Name extraction:
  [Image shows "john smith" written at top]
  → student_name: "John Smith", student_name_confidence: "high"

Example 6 - Multiple Q38s on left side:
  [Image shows Q38 at top-left and another Q38 at bottom-left]
  → q38_left_top: "A", q38_left_bottom: "C" (these are DIFFERENT answers!)

Example 7 - Right side layout:
  [Image shows Q39 and Q40 on the right side]
  → q39_right: "B", q40_right: "D" (remember: right side has NO Q38!)

Example 8 - Faint markings:
  [Image shows very light pencil marks]
  → Use your best judgment; if truly unreadable return "?" with low confidence
"""
    
    if old_examples in content:
        content = content.replace(old_examples, new_examples)
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Added more examples")
        return True
    print("  ✗ Could not find examples section")
    return False


def improve_temperature():
    """Improvement: Lower temperature for more consistent results."""
    print("Applying improvement: Lower Gemini temperature in config...")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "GEMINI_TEMPERATURE = 0.1" in content:
        content = content.replace("GEMINI_TEMPERATURE = 0.1", "GEMINI_TEMPERATURE = 0", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set GEMINI_TEMPERATURE = 0 in config.py")
        return True
    if "GEMINI_TEMPERATURE = 0" in content and "GEMINI_TEMPERATURE = 0.1" not in content:
        content = content.replace("GEMINI_TEMPERATURE = 0", "GEMINI_TEMPERATURE = 0.1", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set GEMINI_TEMPERATURE = 0.1 in config.py")
        return True
    print("  ✗ GEMINI_TEMPERATURE pattern not found")
    return False


def improve_dpi():
    """Improvement: Increase DPI for better image quality."""
    print("Applying improvement: Increase DPI for better image quality...")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    current_dpi = "PDF_DPI = 400"
    new_dpi = "PDF_DPI = 500"
    
    if current_dpi in content:
        content = content.replace(current_dpi, new_dpi)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ Increased DPI to 500")
        return True
    elif "PDF_DPI = 500" in content:
        print("  ✗ DPI already at 500")
        return False
    
    print("  ✗ Could not find DPI setting")
    return False


def improve_crop_region():
    """Improvement: Adjust crop fraction to capture more/less of page."""
    print("Applying improvement: Adjust crop region...")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "CROP_TOP_FRACTION = 0.5" in content:
        content = content.replace("CROP_TOP_FRACTION = 0.5", "CROP_TOP_FRACTION = 0.6", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set CROP_TOP_FRACTION = 0.6 in config.py")
        return True
    if "CROP_TOP_FRACTION = 0.6" in content:
        content = content.replace("CROP_TOP_FRACTION = 0.6", "CROP_TOP_FRACTION = 0.55", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set CROP_TOP_FRACTION = 0.55 in config.py")
        return True
    if "CROP_TOP_FRACTION = 0.55" in content:
        content = content.replace("CROP_TOP_FRACTION = 0.55", "CROP_TOP_FRACTION = 0.6", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set CROP_TOP_FRACTION = 0.6 in config.py")
        return True
    print("  ✗ CROP_TOP_FRACTION pattern not found")
    return False


def improve_model():
    """Improvement: Toggle AI_MODEL between Kimi and Gemini in config."""
    print("Applying improvement: Try different AI_MODEL in config...")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    if 'AI_MODEL = "kimi-k2.5"' in content:
        content = content.replace(
            'AI_MODEL = "kimi-k2.5"',
            'AI_MODEL = "gemini-3.1-pro-preview"',
            1,
        )
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Switched AI_MODEL to gemini-3.1-pro-preview")
        return True
    if 'AI_MODEL = "gemini-3.1-pro-preview"' in content:
        content = content.replace(
            'AI_MODEL = "gemini-3.1-pro-preview"',
            'AI_MODEL = "gemini-3.0-flash"',
            1,
        )
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Switched AI_MODEL to gemini-3.0-flash")
        return True
    if 'AI_MODEL = "gemini-3.0-flash"' in content:
        content = content.replace(
            'AI_MODEL = "gemini-3.0-flash"',
            'AI_MODEL = "kimi-k2.5"',
            1,
        )
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Switched AI_MODEL back to kimi-k2.5")
        return True
    print("  ✗ AI_MODEL pattern not recognized for cycling")
    return False


def improve_max_tokens():
    """Improvement: Bump GEMINI_MAX_OUTPUT_TOKENS in config."""
    print("Applying improvement: Adjust GEMINI_MAX_OUTPUT_TOKENS in config...")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "GEMINI_MAX_OUTPUT_TOKENS = 32000" in content:
        content = content.replace("GEMINI_MAX_OUTPUT_TOKENS = 32000", "GEMINI_MAX_OUTPUT_TOKENS = 65536", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set GEMINI_MAX_OUTPUT_TOKENS = 65536")
        return True
    if "GEMINI_MAX_OUTPUT_TOKENS = 65536" in content:
        content = content.replace("GEMINI_MAX_OUTPUT_TOKENS = 65536", "GEMINI_MAX_OUTPUT_TOKENS = 32000", 1)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✓ Set GEMINI_MAX_OUTPUT_TOKENS = 32000")
        return True
    print("  ✗ GEMINI_MAX_OUTPUT_TOKENS pattern not found")
    return False


def revert_changes():
    """Revert tunable files to last committed state using git."""
    print("Reverting to last known good state...")
    paths = [str(CONFIG_PATH), str(PROFILE_PATH)]
    subprocess.run(["git", "checkout", "--", *paths], check=False)
    print(f"  ✓ Reverted {', '.join(paths)}")


def apply_improvement(strategy: int) -> bool:
    """Apply a specific improvement strategy."""
    strategies = [
        improve_prompt_v1,
        improve_prompt_v2,
        improve_temperature,
        improve_dpi,
        improve_crop_region,
        improve_model,
        improve_max_tokens,
    ]
    
    if 0 <= strategy < len(strategies):
        return strategies[strategy]()
    return False


def main():
    global iteration, best_accuracy, best_config
    
    start_time = time.time()
    max_duration = 4 * 3600  # 4 hours max
    min_duration = 2 * 3600  # 2 hours min
    
    current_phase = 1
    phase_1_complete = False
    
    print("="*60)
    print("IMPROVEMENT AGENT STARTED")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Phase 1: Students {PHASE_1_START}-{PHASE_1_END}")
    print(f"Phase 2: Students {PHASE_2_START}-{PHASE_2_END}")
    print("="*60)
    
    # Initial baseline
    print("\n[BASELINE] Running initial evaluation...")
    result = run_eval(PHASE_1_START, PHASE_1_END)
    best_accuracy = result["accuracy"]
    
    log_event("baseline", {
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "total": result["total"]
    })
    
    print(f"\nBaseline accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
    
    if result["accuracy"] >= TARGET_ACCURACY:
        print("\n🎉 Phase 1 already complete! Moving to Phase 2...")
        phase_1_complete = True
        current_phase = 2
        result = run_eval(PHASE_2_START, PHASE_2_END)
        best_accuracy = result["accuracy"]
    
    errors = analyze_errors(result)
    print(f"\nErrors found: {len(errors)}")
    for e in errors[:5]:
        print(f"  - {e['student']} page {e['page']}: {e['field']}="
              f"'{e['extracted']}' (expected '{e['ground_truth']}')")
    
    # Main improvement loop
    strategy = 0
    no_improvement_count = 0
    
    while True:
        elapsed = time.time() - start_time
        
        # Check time limits
        if elapsed > max_duration:
            print("\n⏰ Maximum duration reached (4 hours). Stopping.")
            break
        
        # Check if we've met the criteria to move to phase 2
        if current_phase == 1 and best_accuracy >= TARGET_ACCURACY and elapsed > min_duration:
            print("\n🎉 Phase 1 complete with 100% accuracy!")
            phase_1_complete = True
            current_phase = 2
            # Reset for phase 2
            best_accuracy = 0.0
            result = run_eval(PHASE_2_START, PHASE_2_END)
            best_accuracy = result["accuracy"]
            print(f"\nPhase 2 baseline: {best_accuracy:.1f}%")
        
        iteration += 1
        
        # Determine which student range to evaluate
        if current_phase == 1:
            eval_start, eval_end = PHASE_1_START, PHASE_1_END
        else:
            eval_start, eval_end = PHASE_2_START, PHASE_2_END
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} (Phase {current_phase}, Strategy {strategy})")
        print(f"Elapsed: {elapsed/60:.1f} minutes")
        print(f"Current best: {best_accuracy:.1f}%")
        print(f"{'='*60}")
        
        # Apply improvement
        improved = apply_improvement(strategy)
        
        if not improved:
            strategy = (strategy + 1) % 7
            if strategy == 0:
                print("\n⚠️ All strategies tried without code changes. Reverting...")
                revert_changes()
            continue
        
        # Run evaluation
        result = run_eval(eval_start, eval_end)
        new_accuracy = result["accuracy"]
        
        log_event("iteration", {
            "phase": current_phase,
            "strategy": strategy,
            "accuracy": new_accuracy,
            "correct": result["correct"],
            "total": result["total"],
            "improved": new_accuracy > best_accuracy
        })
        
        if new_accuracy > best_accuracy:
            print(f"\n✅ IMPROVEMENT! {best_accuracy:.1f}% → {new_accuracy:.1f}%")
            best_accuracy = new_accuracy
            best_config = strategy
            no_improvement_count = 0
            
            # Save the improved version
            subprocess.run(["git", "add", str(CONFIG_PATH), str(PROFILE_PATH)], check=False)
            subprocess.run(["git", "commit", "-m", 
                          f"Improvement iteration {iteration}: strategy {strategy}, accuracy {best_accuracy:.1f}%"], 
                         check=False)
            
            # Analyze remaining errors
            errors = analyze_errors(result)
            if errors:
                print(f"\nRemaining errors ({len(errors)}):")
                for e in errors[:5]:
                    print(f"  - {e['student']} page {e['page']}: {e['field']}="
                          f"'{e['extracted']}' (expected '{e['ground_truth']}')")
        else:
            print(f"\n❌ No improvement: {new_accuracy:.1f}% (best: {best_accuracy:.1f}%)")
            no_improvement_count += 1
            
            # Revert if no improvement
            revert_changes()
            
            # If we've tried many strategies without improvement, take a break
            if no_improvement_count >= 3:
                print("\n😴 No improvement for 3 iterations. Taking a short break...")
                time.sleep(30)
                no_improvement_count = 0
        
        # Move to next strategy
        strategy = (strategy + 1) % 7
        
        # Short delay between iterations
        time.sleep(5)
    
    # Final summary
    print("\n" + "="*60)
    print("IMPROVEMENT AGENT COMPLETED")
    print(f"Total iterations: {iteration}")
    print(f"Final best accuracy: {best_accuracy:.1f}%")
    print(f"Phase 1 complete: {phase_1_complete}")
    print(f"Time elapsed: {(time.time() - start_time)/60:.1f} minutes")
    print("="*60)


if __name__ == "__main__":
    main()
