"""
src/antonia/kb/store.py

ChromaStore — read-only runtime retrieval interface.
No ingestion methods; write path lives in kb/ingest/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from antonia.config.settings import KBConfig
from antonia.domain.utterance import KnowledgeChunk, RetrievalContext

log = structlog.get_logger(__name__)

_COLLECTION_NAME = "antonia_lab"


class ChromaStore:
    def __init__(self, config: KBConfig) -> None:
        self._cfg = config
        self._client: Any = None
        self._collection: Any = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            import chromadb

            if self._cfg.mode == "server":
                self._client = chromadb.HttpClient(
                    host=self._cfg.host,
                    port=self._cfg.port,
                )
            else:
                persist_path = Path(self._cfg.persist_directory)
                if not persist_path.exists():
                    log.info("chroma_store_missing", path=str(persist_path))
                    return
                self._client = chromadb.PersistentClient(path=str(persist_path))

            collections = [c.name for c in self._client.list_collections()]
            if _COLLECTION_NAME in collections:
                self._collection = self._client.get_collection(_COLLECTION_NAME)
                self._available = True
                log.info("chroma_connected", mode=self._cfg.mode)
            else:
                log.info("chroma_collection_missing", name=_COLLECTION_NAME)
        except Exception as exc:
            log.warning("chroma_connect_failed", error=str(exc))

    def is_available(self) -> bool:
        return self._available

    def query_raw(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        if not self._available or self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_texts=[text],
                n_results=min(n_results, self._collection.count()),
            )
            return results
        except Exception as exc:
            log.warning("chroma_query_error", error=str(exc))
            return []


class KBRetriever:
    def __init__(self, store: ChromaStore, config: KBConfig) -> None:
        self._store = store
        self._threshold = config.relevance_threshold

    def is_available(self) -> bool:
        return self._store.is_available()

    def query(self, text: str, n_results: int = 3) -> RetrievalContext:
        if not self._store.is_available():
            return RetrievalContext(chunks=(), query=text)

        raw = self._store.query_raw(text, n_results)
        if not raw:
            return RetrievalContext(chunks=(), query=text)

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        chunks: list[KnowledgeChunk] = []
        seen_sources: set[str] = set()

        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - float(dist)
            if score < self._threshold:
                continue
            source = str(meta.get("source", "unknown"))
            if source in seen_sources:
                continue
            seen_sources.add(source)
            chunks.append(KnowledgeChunk(text=str(doc), source=source, score=score))

        log.debug("kb_retrieved", n=len(chunks), query_len=len(text))
        return RetrievalContext(chunks=tuple(chunks), query=text)


class MockKBRetriever:
    def is_available(self) -> bool:
        return False

    def query(self, text: str, n_results: int = 3) -> RetrievalContext:
        return RetrievalContext(chunks=(), query=text)
