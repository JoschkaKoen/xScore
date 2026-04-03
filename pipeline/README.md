# Pipeline package layout

The grading pipeline is split into **domain subpackages** under `pipeline/`. [`grade.py`](../grade.py) wires them in order (mostly via late imports in `_run`).

## Subpackages

| Folder | Role |
|--------|------|
| [`preprocessing/`](preprocessing/) | Raw class scan → `cleaned_scan.pdf` (blank removal, autorotate, deskew, optional debug PDFs). |
| [`scaffold/`](scaffold/) | Vector exam + answer key → `ExamScaffold`, cache, figure PNGs, boxes on empty exam, geometry onto scans. |
| [`marking/`](marking/) | Kimi-driven steps: parse instruction, find folder, assign pages, detect attempted questions, grade. |
| [`reports/`](reports/) | Terminal tables / summaries and LaTeX → PDF report. |
| [`shared/`](shared/) | Dataclasses, path helpers, CLI formatting, roster and ground-truth I/O. |

## `grade.py` step → module map

| Step | Module (import path) | Notes |
|------|----------------------|--------|
| 1 | `pipeline.marking.parse_instruction` | `parse_prompt(...)` |
| 2 | `pipeline.marking.find_exam_folder` | `find_folder(...)` |
| 3 | `pipeline.shared.load_student_list` | `read_student_list(...)` |
| 4 | `pipeline.scaffold.generate_scaffold` | `build_scaffold(...)` |
| 5 | `pipeline.preprocessing.start_scan` | `cleanup_pdf(...)` |
| 6 | `pipeline.marking.assign_pages_to_students` | `assign_pages(...)` |
| 7 | `pipeline.marking.detect_answered_questions` | `detect_answered_exercises(...)` |
| 8 | `pipeline.marking.grade_answers` | `grade_students(...)` |
| 9 | `pipeline.reports.print_results` | `print_*` helpers |
| 10 | `pipeline.shared.load_ground_truth` | optional evaluation |
| 11 | `pipeline.reports.generate_report` | `generate_report(...)` |

## Vector PDF parsing

[`scaffold/pdf_parser/`](scaffold/pdf_parser/) implements layout detection, regions, content extraction, and assembly into `Question` trees. Import the stable surface from `pipeline.scaffold.pdf_parser` (same symbols as before the move).
