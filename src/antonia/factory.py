"""
src/antonia/factory.py

Factory functions that instantiate the correct backend implementations
based on AntoniaSettings. The orchestrator and scripts only call these —
they never import concrete backend classes directly.
"""

from __future__ import annotations

import httpx
import structlog

from antonia.config.settings import AntoniaSettings

log = structlog.get_logger(__name__)


def build_stt(settings: AntoniaSettings) -> object:
    backend = settings.stt.backend
    if backend == "mock":
        from antonia.stt.mock import MockSTTBackend
        return MockSTTBackend()
    from antonia.stt.whisper import WhisperBackend
    return WhisperBackend(
        config=settings.stt,
        sr_hw=settings.audio.sample_rate_hw,
        gain=settings.audio.gain,
    )


def build_llm(settings: AntoniaSettings, client: httpx.AsyncClient) -> object:
    if settings.llm.backend == "mock":
        from antonia.llm.mock import MockLLMBackend
        return MockLLMBackend()
    from antonia.llm.ollama import OllamaBackend
    return OllamaBackend(config=settings.llm, client=client)


def build_tts(settings: AntoniaSettings) -> object:
    from antonia.tts.preprocessor import TextPreprocessor

    pre = TextPreprocessor(settings.phonetic_map_path)

    if settings.tts.backend == "mock":
        from antonia.tts.mock import MockTTSBackend
        return MockTTSBackend()

    from antonia.tts.backend import TTSBackend

    kokoro = None
    piper = None

    kokoro_exc: Exception | None = None
    piper_exc: Exception | None = None

    if settings.tts.backend in ("kokoro-cuda", "kokoro-cpu"):
        try:
            from antonia.tts.kokoro import KokoroBackend
            kokoro = KokoroBackend(
                model_path=settings.kokoro_model,
                voices_path=settings.kokoro_voices,
                onnx_providers=settings.tts.onnx_providers,
            )
        except (FileNotFoundError, ImportError) as exc:
            kokoro_exc = exc
            log.warning("kokoro_unavailable", error=str(exc))

    try:
        from antonia.tts.piper import PiperBackend
        piper = PiperBackend(
            es_model=settings.piper_es_model,
            en_model=settings.piper_en_model,
        )
    except (FileNotFoundError, ImportError) as exc:
        piper_exc = exc
        log.warning("piper_unavailable", error=str(exc))

    if kokoro is None and piper is None:
        causes = []
        if kokoro_exc:
            causes.append(f"kokoro: {kokoro_exc}")
        if piper_exc:
            causes.append(f"piper: {piper_exc}")
        raise RuntimeError(
            "No TTS engine could be loaded. " + " | ".join(causes)
        )

    return TTSBackend(
        preprocessor=pre,
        kokoro=kokoro,
        piper=piper,
        output_dir=settings.base_dir / "tests" / "pipeline_runs",
    )


def build_relay(settings: AntoniaSettings, stt: object) -> object:
    if not settings.gpu_relay.enabled:
        from antonia.pipeline.relay import NullRelay
        return NullRelay()

    from antonia.infra.memory import TorchMemoryManager
    from antonia.pipeline.relay import GPURelay

    mem = TorchMemoryManager(
        target_mb=settings.gpu_relay.unload_target_mb,
        poll_interval_s=settings.gpu_relay.poll_interval_seconds,
        max_polls=settings.gpu_relay.max_polls,
    )
    return GPURelay(stt=stt, memory=mem)


def build_kb(settings: AntoniaSettings) -> object | None:
    try:
        from antonia.kb.store import ChromaStore, KBRetriever
        store = ChromaStore(settings.kb)
        if not store.is_available():
            log.info("kb_not_available")
            return None
        return KBRetriever(store=store, config=settings.kb)
    except ImportError:
        log.warning("chromadb_missing")
        return None


def build_prompt_template(settings: AntoniaSettings) -> object:
    from antonia.domain.prompt import PromptRegistry
    registry = PromptRegistry(settings.prompt_dir)
    return registry.get(settings.prompt_name)
