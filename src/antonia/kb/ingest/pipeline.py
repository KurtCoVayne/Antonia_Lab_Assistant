"""
src/antonia/kb/ingest/pipeline.py

Offline ingestion pipeline — run via `antonia-ingest`, never at inference time.
Reads raw documents → chunks → embeds via Ollama → writes to Chroma.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import structlog

from antonia.config.settings import AntoniaSettings

log = structlog.get_logger(__name__)

_COLLECTION_NAME = "antonia_lab"
_HASH_KEY = "corpus_hash"


def _hash_directory(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _read_documents(raw_dir: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in (".txt", ".md"):
            docs.append({"text": path.read_text(encoding="utf-8"), "source": path.name})
        elif path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
                docs.append({"text": text, "source": path.name})
            except ImportError:
                log.warning("pypdf_missing", path=str(path))
    log.info("documents_read", count=len(docs))
    return docs


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _embed(texts: list[str], model: str, ollama_url: str) -> list[list[float]]:
    embeddings: list[list[float]] = []
    with httpx.Client(timeout=60.0) as client:
        for text in texts:
            resp = client.post(
                f"{ollama_url}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings.append(data["embeddings"][0])
    return embeddings


def run_ingestion(settings: AntoniaSettings) -> None:
    raw_dir = settings.base_dir / "knowledge_base" / "raw"
    if not raw_dir.exists():
        log.warning("raw_dir_missing", path=str(raw_dir))
        return

    corpus_hash = _hash_directory(raw_dir)
    log.info("ingestion_start", corpus_hash=corpus_hash)

    import chromadb

    persist_path = settings.kb_persist_path
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))

    existing = [c.name for c in client.list_collections()]
    if _COLLECTION_NAME in existing:
        col = client.get_collection(_COLLECTION_NAME)
        stored_hash = col.metadata.get(_HASH_KEY, "")
        if stored_hash == corpus_hash:
            log.info("ingestion_skipped", reason="corpus unchanged")
            return
        client.delete_collection(_COLLECTION_NAME)

    col = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={_HASH_KEY: corpus_hash},
    )

    docs = _read_documents(raw_dir)
    if not docs:
        log.warning("ingestion_no_documents")
        return

    all_chunks: list[str] = []
    all_sources: list[str] = []
    for doc in docs:
        chunks = _chunk_text(doc["text"], settings.kb.chunk_size, settings.kb.chunk_overlap)
        all_chunks.extend(chunks)
        all_sources.extend([doc["source"]] * len(chunks))

    log.info("ingestion_embedding", n_chunks=len(all_chunks))
    embeddings = _embed(all_chunks, settings.kb.embedding_model, settings.llm.base_url)

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas: list[dict[str, Any]] = [{"source": s} for s in all_sources]

    batch = 100
    for i in range(0, len(all_chunks), batch):
        col.add(
            ids=ids[i : i + batch],
            documents=all_chunks[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    log.info("ingestion_complete", n_chunks=len(all_chunks), persist=str(persist_path))
