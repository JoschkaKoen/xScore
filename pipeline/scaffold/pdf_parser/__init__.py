"""Deterministic vector-PDF parsing for exam papers and answer keys (PyMuPDF).

Layout and margin anchors live in :mod:`pipeline.scaffold.pdf_parser.layout` and
:mod:`pipeline.scaffold.pdf_parser.regions`; stem heuristics in :mod:`pipeline.scaffold.pdf_parser.content`;
Cambridge-style sub-parts in :mod:`pipeline.scaffold.pdf_parser.subparts`; assembly in
:mod:`pipeline.scaffold.pdf_parser.build`. Import from ``pipeline.pdf_parser`` for the stable
public API below.
"""

from __future__ import annotations

from pipeline.scaffold.pdf_parser.api import (
    merge_answers_into_scaffold,
    parse_answer_key_pdf,
    parse_exam_pdf,
    prepare_scaffold_image_dirs,
)
from pipeline.scaffold.pdf_parser.config import DEFAULT_PARSER_CONFIG, ParserConfig
from pipeline.scaffold.pdf_parser.layout import page_layout_cells
from pipeline.scaffold.pdf_parser.regions import find_question_positions, iter_region_segments

__all__ = [
    "DEFAULT_PARSER_CONFIG",
    "ParserConfig",
    "find_question_positions",
    "iter_region_segments",
    "merge_answers_into_scaffold",
    "page_layout_cells",
    "parse_answer_key_pdf",
    "parse_exam_pdf",
    "prepare_scaffold_image_dirs",
]
