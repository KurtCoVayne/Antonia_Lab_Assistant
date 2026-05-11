"""
src/antonia/config/settings.py

AntoniaSettings — single source of truth for all runtime configuration.
Loaded once at startup via AntoniaSettings.load(). All hardware-specific
constants come from the active profile YAML; no module may contain
hardcoded device strings, model paths, or numeric tuning parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Profile selection ──────────────────────────────────────────────────────

_PROFILE_DIR = Path(__file__).parent / "profiles"

ProfileName = Literal["jetson", "apple-silicon", "cpu-only"]

_PROFILE_FILES: dict[str, str] = {
    "jetson":          "jetson.yaml",
    "apple-silicon":   "apple_silicon.yaml",
    "cpu-only":        "cpu_only.yaml",
}


def _load_yaml_profile(name: str) -> dict[str, Any]:
    filename = _PROFILE_FILES.get(name)
    if filename is None:
        raise ValueError(
            f"Unknown ANTONIA_PROFILE '{name}'. "
            f"Valid values: {list(_PROFILE_FILES)}"
        )
    path = _PROFILE_DIR / filename
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


# ── Sub-models ─────────────────────────────────────────────────────────────

class STTConfig(BaseModel):
    backend: Literal["whisper-cuda", "whisper-cpu", "mock"] = "whisper-cpu"
    compute_type: str = "float32"
    device: str = "cpu"
    device_index: int = 0
    num_workers: int = 2
    cpu_threads: int = 4
    model_size: str = "small"


class LLMConfig(BaseModel):
    backend: Literal["ollama", "mock"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:1.5b"
    keep_alive_seconds: int = 30
    num_ctx: int = 2048
    temperature: float = 0.1
    num_predict: int = 200


class TTSConfig(BaseModel):
    backend: Literal["kokoro-cuda", "kokoro-cpu", "piper", "mock"] = "kokoro-cpu"
    onnx_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])


class AudioConfig(BaseModel):
    device_index: int | None = None
    sample_rate_hw: int = 44100
    chunk_samples: int = 4410
    gain: float = 3.0


class GPURelayConfig(BaseModel):
    enabled: bool = False
    unload_target_mb: float = 50.0
    poll_interval_seconds: float = 0.05
    max_polls: int = 20


class KBConfig(BaseModel):
    mode: Literal["embedded", "server"] = "embedded"
    host: str = "localhost"
    port: int = 8000
    persist_directory: str = "knowledge_base/chroma"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 512
    chunk_overlap: int = 64
    relevance_threshold: float = 0.72


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["console", "json"] = "console"


# ── Root settings ──────────────────────────────────────────────────────────

class AntoniaSettings(BaseSettings):
    """
    All configuration for the Antonia pipeline.

    Environment variables:
        ANTONIA_PROFILE  — profile name (jetson | apple-silicon | cpu-only)
        ANTONIA_BASE_DIR — absolute path to the repo root (default: auto-detected)
        ANTONIA_MODELS_DIR — path to models/ directory (default: <base_dir>/models)
    """

    model_config = SettingsConfigDict(
        env_prefix="ANTONIA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    profile: ProfileName = "apple-silicon"
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parents[3].resolve())
    models_dir: Path | None = None

    # Sub-models populated from profile YAML; may be overridden by env vars.
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    gpu_relay: GPURelayConfig = Field(default_factory=GPURelayConfig)
    kb: KBConfig = Field(default_factory=KBConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Wakeword constants (stable across profiles)
    wakeword_threshold: float = 0.35
    wakeword_flush_chunks: int = 6
    vad_threshold: float = 0.40
    silence_windows: int = 20
    min_speech_windows: int = 3
    noise_gate_rms: float = 0.008
    listening_timeout_chunks: int = 60

    # Conversation history window (max messages kept)
    history_max_messages: int = 4

    # Prompt registry
    prompt_name: str = "laboratory"

    @model_validator(mode="before")
    @classmethod
    def _merge_profile(cls, values: dict[str, Any]) -> dict[str, Any]:
        profile_name = str(values.get("profile", os.environ.get("ANTONIA_PROFILE", "apple-silicon")))
        try:
            profile = _load_yaml_profile(profile_name)
        except (ValueError, FileNotFoundError):
            return values

        # Profile values are defaults; explicit dict keys (from env vars) win.
        for section in ("stt", "llm", "tts", "audio", "gpu_relay", "kb", "logging"):
            if section in profile and section not in values:
                values[section] = profile[section]

        return values

    @property
    def models_path(self) -> Path:
        return self.models_dir or (self.base_dir / "models")

    @property
    def kokoro_model(self) -> Path:
        return self.models_path / "kokoro" / "kokoro-v1.0.onnx"

    @property
    def kokoro_voices(self) -> Path:
        return self.models_path / "kokoro" / "voices-v1.0.bin"

    @property
    def piper_es_model(self) -> Path:
        return self.models_path / "piper" / "es_MX-claude-high.onnx"

    @property
    def piper_en_model(self) -> Path:
        return self.models_path / "piper" / "en_US-lessac-medium.onnx"

    @property
    def phonetic_map_path(self) -> Path:
        return self.base_dir / "knowledge_base" / "phonetic_map.yaml"

    @property
    def kb_persist_path(self) -> Path:
        p = Path(self.kb.persist_directory)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    @property
    def prompt_dir(self) -> Path:
        return self.base_dir / "config" / "prompts"

    @classmethod
    def load(cls) -> "AntoniaSettings":
        return cls()


# Module-level singleton — created once when the package is first imported.
# Tests override this by patching antonia.config.settings.settings directly.
settings: AntoniaSettings = AntoniaSettings.load()
