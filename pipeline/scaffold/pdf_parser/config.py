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
    # Pixmap clip = core rect expanded by this margin on all sides (exam + answer key).
    image_crop_pad_pt: float = 5.0
    # When the PDF has no embedded raster images, rasterize vector drawings (e.g. Fig. n.n).
    vector_figure_fallback: bool = True
    vector_figure_min_area: float = 400.0
    vector_figure_min_short_side: float = 20.0
    vector_figure_max_aspect: float = 6.0
    # Union vector bbox with nearby printed labels (A/B, Sun, “Fig. n.n”) before rasterizing.
    vector_figure_label_h_pad_pt: float = 32.0
    vector_figure_label_v_pad_top_pt: float = 10.0
    vector_figure_label_v_pad_bottom_pt: float = 10.0
    vector_figure_label_max_line_chars: int = 56
    # Snap bboxes to layout cell (sub-page) top/bottom when near the edge.
    subpage_edge_snap_tol_top_pt: float = 20.0
    subpage_edge_snap_tol_bottom_pt: float = 24.0
    # Gap between the previous exercise's last text line and the next leaf bbox top.
    leaf_bbox_gap_after_previous_line_pt: float = 2.0
    # Equation-blank bbox geometry (for "label = …… [n]" answer lines).
    equation_blank_pad_above_pt: float = 30.0
    equation_blank_pad_below_subpage_pt: float = 25.0
    equation_blank_subpage_bottom_tol_pt: float = 40.0
    equation_blank_nudge_top_pt: float = 5.0
    equation_blank_nudge_bottom_pt: float = 5.0


DEFAULT_PARSER_CONFIG = ParserConfig()
