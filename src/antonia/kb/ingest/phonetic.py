"""
src/antonia/kb/ingest/phonetic.py

PhoneticMap builder — extracts technical terms from ingested documents
and appends entries to knowledge_base/phonetic_map.yaml.
Run as part of the offline ingestion pipeline, not at inference time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)

# Heuristic patterns for technical terms worth adding to the phonetic map
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}(?:-[A-Z0-9]+)?\b")
_MODEL_RE    = re.compile(r"\b[A-Z][0-9]+(?:-[A-Z0-9]+)+\b")


def extract_terms(texts: list[str]) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        terms.update(_ACRONYM_RE.findall(text))
        terms.update(_MODEL_RE.findall(text))
    return terms


def update_phonetic_map(
    terms: set[str],
    phonetic_map_path: Path,
) -> None:
    if not phonetic_map_path.exists():
        log.warning("phonetic_map_missing", path=str(phonetic_map_path))
        return

    with phonetic_map_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    existing_terms: dict[str, str] = data.get("terms", {})
    added = 0

    for term in sorted(terms):
        pattern = f"\\b{re.escape(term)}\\b"
        if pattern not in existing_terms:
            # Default phonetic: spell out the term letter by letter
            phonetic = "-".join(term.upper())
            existing_terms[pattern] = phonetic
            added += 1

    data["terms"] = existing_terms
    with phonetic_map_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    log.info("phonetic_map_updated", added=added, total=len(existing_terms))
