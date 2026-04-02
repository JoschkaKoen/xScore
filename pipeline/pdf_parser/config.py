"""Tunable parameters for vector PDF exam parsing."""

from __future__ import annotations

from dataclasses import dataclass

# Cambridge-style footer / disclaimer trimming (see regions.trim)
DISCLAIMER_TRIGGERS = (
    "permission to reproduce",
    "third-party owned material",
    "copyright acknowledgements booklet",
)
MIN_TRIM_GAP_PT = 4.0
FOOTER_MARGIN_PT = 15.0
SEPARATOR_MIN_WIDTH_PT = 100.0

# Nominal A4 size in PDF points (scale margins for imposed / mini pages).
NOMINAL_A4_W = 595.276
NOMINAL_A4_H = 841.89
FOURUP_PORTRAIT_MIN_W = 700.0
FOURUP_PORTRAIT_MIN_H = 950.0
A3_LANDSCAPE_MIN_W = 1000.0
A3_LANDSCAPE_MIN_H = 600.0


@dataclass(frozen=True)
class ParserConfig:
    margin_top: float = 55.0
    margin_bottom: float = 790.0
    anchor_margin_top: float = 10.0
    question_x_max: float = 60.0
    font_size_min: float = 5.0
    font_size_max: float = 14.0
    padding_above: float = 8.0
    text_clip_pad_above_pt: float = 2.0
    strip_crop_left: float = 45.0
    strip_crop_right: float = 22.0
    box_min_width: float = 80.0
    box_min_height: float = 15.0
    line_max_height: float = 4.0
    line_min_width: float = 100.0


DEFAULT_PARSER_CONFIG = ParserConfig()
