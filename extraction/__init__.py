"""Answer extraction package: profiles, providers, reporting."""

from __future__ import annotations

__version__ = "0.1"

from extraction.providers import call_ocr_api, create_extraction_client, get_provider, multi_pass_extract

__all__ = [
    "__version__",
    "call_ocr_api",
    "create_extraction_client",
    "get_provider",
    "multi_pass_extract",
]
