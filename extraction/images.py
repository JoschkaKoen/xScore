"""Image crop, preprocess, JPEG encoding, and MC answer normalization."""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image, ImageEnhance

from config import (
    CROP_TOP_FRACTION,
    JPEG_QUALITY,
    PREPROCESS_BRIGHTNESS,
    PREPROCESS_CONTRAST,
    PREPROCESS_SHARPNESS,
)


def effective_crop_fraction() -> float:
    """Optional override via ``EXTRACT_CROP_FRACTION`` (e.g. 0.55) for crop tuning."""
    raw = os.getenv("EXTRACT_CROP_FRACTION", "").strip()
    if not raw:
        return CROP_TOP_FRACTION
    try:
        v = float(raw)
        return max(0.2, min(1.0, v))
    except ValueError:
        return CROP_TOP_FRACTION


def crop_top(image: Image.Image, fraction: float = CROP_TOP_FRACTION) -> Image.Image:
    """Return the top `fraction` of the image."""
    w, h = image.size
    return image.crop((0, 0, w, int(h * fraction)))


def preprocess_for_extraction(image: Image.Image) -> Image.Image:
    """Enhance image for better handwriting recognition."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(PREPROCESS_CONTRAST)

    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(PREPROCESS_SHARPNESS)

    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(PREPROCESS_BRIGHTNESS)

    return image


def to_jpeg_bytes(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    """Convert a PIL image to JPEG bytes."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def normalize_mc_answer(val: Any) -> str:
    """Coerce model output to a single ``A``/``B``/``C``/``D`` or ``?``."""
    if val is None:
        return "?"
    s = str(val).upper().strip()
    if not s or s == "?":
        return "?"
    letters = [c for c in s if c in "ABCD"]
    if not letters:
        return "?"
    if len(set(letters)) > 1:
        return "?"
    return letters[0]


def normalize_extracted_record(data: dict, answer_fields: list[str]) -> dict:
    """Normalize all MC answer fields in place (and return ``data``)."""
    for field in answer_fields:
        if field in data:
            data[field] = normalize_mc_answer(data.get(field))
    return data
