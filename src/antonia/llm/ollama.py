"""
src/antonia/llm/ollama.py

OllamaBackend — async httpx client for the Ollama API.
R-4: Exponential backoff with jitter on OOM errors.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncGenerator

import httpx
import structlog

from antonia.config.settings import LLMConfig
from antonia.domain.utterance import LLMResponse

log = structlog.get_logger(__name__)

_MAX_RETRIES = 2


class OllamaBackend:
    def __init__(self, config: LLMConfig, client: httpx.AsyncClient) -> None:
        self._cfg = config
        self._client = client
        self._url = f"{config.base_url}/api/chat"

    async def ask(
        self,
        messages: list[dict[str, str]],
        retry: int = 0,
    ) -> LLMResponse:
        payload = {
            "model":      self._cfg.model,
            "keep_alive": self._cfg.keep_alive_seconds,
            "stream":     False,
            "options": {
                "num_ctx":        self._cfg.num_ctx,
                "temperature":    self._cfg.temperature,
                "top_p":          0.8,
                "repeat_penalty": 1.15,
                "num_predict":    self._cfg.num_predict,
            },
            "messages": messages,
        }

        t0 = time.time()
        try:
            resp = await self._client.post(self._url, json=payload)
            data = resp.json()
            elapsed = time.time() - t0

            if "error" in data:
                err = str(data["error"])
                log.error("ollama_error", error=err)
                if "out of memory" in err and retry < _MAX_RETRIES:
                    delay = (2**retry) + random.uniform(0.0, 1.0)
                    log.warning("ollama_oom_retry", delay_s=round(delay, 2), attempt=retry + 1)
                    await asyncio.sleep(delay)
                    return await self.ask(messages, retry + 1)
                return LLMResponse(text="", latency_s=elapsed)

            eval_c = data.get("eval_count", 0)
            eval_d = data.get("eval_duration", 1) / 1e9
            tok_s = eval_c / eval_d if eval_d > 0 else 0.0
            text = data.get("message", {}).get("content", "").strip()

            log.debug(
                "llm_response",
                latency_s=round(elapsed, 3),
                tokens_per_sec=round(tok_s, 1),
                text_len=len(text),
            )
            return LLMResponse(text=text, latency_s=elapsed, tokens_per_sec=tok_s)

        except httpx.TimeoutException:
            elapsed = time.time() - t0
            log.error("ollama_timeout", elapsed_s=round(elapsed, 1))
            return LLMResponse(text="", latency_s=elapsed)
        except Exception as exc:
            elapsed = time.time() - t0
            log.error("ollama_unexpected", error=str(exc))
            return LLMResponse(text="", latency_s=elapsed)

    async def ask_stream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model":      self._cfg.model,
            "keep_alive": self._cfg.keep_alive_seconds,
            "stream":     True,
            "options": {
                "num_ctx":        self._cfg.num_ctx,
                "temperature":    self._cfg.temperature,
                "top_p":          0.8,
                "repeat_penalty": 1.15,
                "num_predict":    self._cfg.num_predict,
            },
            "messages": messages,
        }
        try:
            async with self._client.stream("POST", self._url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in data:
                        log.error("ollama_stream_error", error=str(data["error"]))
                        return
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        return
        except httpx.TimeoutException:
            log.error("ollama_stream_timeout")
        except Exception as exc:
            log.error("ollama_stream_unexpected", error=str(exc))
