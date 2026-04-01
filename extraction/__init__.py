"""Answer extraction package: profiles, providers, eval, reporting."""

from __future__ import annotations

from version import __version__

from extraction.eval import extract_first_n_students_eval
from extraction.providers import call_ocr_api, create_extraction_client, get_provider, multi_pass_extract

__all__ = [
    "__version__",
    "call_ocr_api",
    "create_extraction_client",
    "extract_first_n_students_eval",
    "get_provider",
    "multi_pass_extract",
]
