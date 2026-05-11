"""
src/antonia/infra/health.py

Startup health checks for external services.
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger(__name__)


async def check_ollama(base_url: str, expected_model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url}/api/tags")
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        log.info("ollama_available", models=models)
        model_base = expected_model.split(":")[0]
        if not any(model_base in m for m in models):
            log.warning("ollama_model_missing", model=expected_model)
            return False
        return True
    except Exception as exc:
        log.error("ollama_unreachable", url=base_url, error=str(exc))
        return False


async def check_chroma(host: str, port: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"http://{host}:{port}/api/v1/heartbeat")
        r.raise_for_status()
        log.info("chroma_available", host=host, port=port)
        return True
    except Exception as exc:
        log.warning("chroma_unreachable", host=host, port=port, error=str(exc))
        return False
