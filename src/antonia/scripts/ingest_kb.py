"""
src/antonia/scripts/ingest_kb.py

Entry point: antonia-ingest
Runs the offline KB ingestion pipeline and updates the phonetic map.
"""

from __future__ import annotations


def main() -> None:
    from antonia.config.settings import settings
    from antonia.infra.logging import configure_logging
    from antonia.kb.ingest.pipeline import run_ingestion
    from antonia.kb.ingest.phonetic import extract_terms, update_phonetic_map

    configure_logging(level=settings.logging.level, fmt=settings.logging.format)

    import structlog
    log = structlog.get_logger(__name__)
    log.info("ingest_start")

    run_ingestion(settings)

    raw_dir = settings.base_dir / "knowledge_base" / "raw"
    if raw_dir.exists():
        texts = [
            p.read_text(encoding="utf-8", errors="replace")
            for p in raw_dir.rglob("*.txt")
        ] + [
            p.read_text(encoding="utf-8", errors="replace")
            for p in raw_dir.rglob("*.md")
        ]
        if texts:
            terms = extract_terms(texts)
            update_phonetic_map(terms, settings.phonetic_map_path)

    log.info("ingest_done")


if __name__ == "__main__":
    main()
