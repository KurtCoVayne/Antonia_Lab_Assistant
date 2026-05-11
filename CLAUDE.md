# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Antonia is a voice-controlled AI assistant for the Digital Control Laboratory at Universidad EAFIT (Medellín, Colombia). It runs on a **NVIDIA Jetson Orin Nano 8GB** and is developed on **Apple Silicon M4**. Pipeline: Wake Word → STT → KB Retrieval → LLM → TTS.

All library code lives in `src/antonia/`. Platform-specific constants live in `src/antonia/config/profiles/*.yaml`.

---

## Running the Project

```bash
# Create venv and install (first time)
uv venv .venv --python 3.11
uv pip install -e ".[dev]"

# Run full pipeline (N_TURNS cycles with manual recording)
ANTONIA_PROFILE=apple-silicon antonia-run

# Ingest knowledge base documents
antonia-ingest

# Benchmark STT latency
antonia-bench

# Tests
make test          # unit tests (no hardware)
make test-live     # integration tests (requires Ollama)

# Lint + typecheck
make lint
make typecheck

# Start infrastructure services (Ollama + ChromaDB)
make dev-up        # Mac M4
make jetson-up     # Jetson (adds --network host + NVIDIA runtime)
make down
```

---

## Platform Selection

Set `ANTONIA_PROFILE` environment variable (or let it default to `apple-silicon`):

| Profile | Hardware | STT | GPU Relay | TTS |
|---|---|---|---|---|
| `apple-silicon` | Mac M4 | Whisper CPU float32 | disabled | Kokoro CPU |
| `jetson` | Jetson Orin Nano | Whisper CUDA float16 | **enabled** | Kokoro CUDA |
| `cpu-only` | Any CPU | Whisper CPU int8 | disabled | Piper |

No code changes are needed when switching platforms — only the env var changes.

---

## Architecture

### Critical Constraint — GPU Relay (Jetson only)

Jetson has **unified memory** (~8 GB shared CPU+GPU). Whisper and Qwen cannot coexist in VRAM. The relay is managed by `GPURelay` (`src/antonia/pipeline/relay.py`):

```
STT: Whisper on GPU (~900 MB)
  → relay.before_llm()   # unload_gpu() + drop_os_cache() in parallel
LLM: Qwen on GPU (~1600 MB)
  → KEEP_ALIVE=0 auto-unloads Qwen
TTS: Kokoro on CPU/GPU
  → relay.after_tts()    # reload_gpu()
```

On Mac M4 and `cpu-only`, `NullRelay` is used — both methods are no-ops.

### Directory Map

```
src/antonia/
├── config/
│   ├── settings.py          # AntoniaSettings (pydantic-settings)
│   └── profiles/            # jetson.yaml | apple_silicon.yaml | cpu_only.yaml
├── domain/
│   ├── utterance.py         # RawAudio, TranscriptionResult, LLMResponse, SynthesisResult
│   └── prompt.py            # SystemPrompt, PromptTemplate, PromptRegistry
├── audio/
│   ├── dsp.py               # resample, preemphasis, peak_normalize (no model deps)
│   ├── vad.py               # SileroVAD, VadBuffer (pre-allocated)
│   └── capture.py           # AudioCapture (sounddevice abstraction)
├── stt/
│   ├── whisper.py           # WhisperBackend (faster-whisper, GPU lifecycle)
│   └── mock.py
├── llm/
│   ├── ollama.py            # OllamaBackend (httpx async, exponential backoff)
│   └── mock.py
├── tts/
│   ├── preprocessor.py      # TextPreprocessor (markdown strip, units, phonetics)
│   ├── kokoro.py            # KokoroBackend (ONNX, CUDA fallback H-3)
│   ├── piper.py             # PiperBackend (warm-pool subprocess R-3)
│   ├── backend.py           # TTSBackend (facade)
│   └── mock.py
├── kb/
│   ├── store.py             # ChromaStore (read-only), KBRetriever
│   └── ingest/
│       ├── pipeline.py      # Offline: docs → chunks → embed → Chroma
│       └── phonetic.py      # Extracts technical terms → phonetic_map.yaml
├── wakeword/
│   ├── openwakeword.py      # OpenWakeWordBackend (hey_jarvis)
│   └── mock.py
├── pipeline/
│   ├── orchestrator.py      # AntoniaOrchestrator (async state machine)
│   ├── relay.py             # GPURelay / NullRelay
│   └── context.py           # ConversationContext (typed history)
├── infra/
│   ├── logging.py           # structlog (console | json)
│   ├── memory.py            # TorchMemoryManager / NullMemoryManager
│   └── health.py            # check_ollama(), check_chroma()
├── factory.py               # build_stt/llm/tts/relay() — selects concrete backend
└── scripts/
    ├── run_antonia.py       # antonia-run
    ├── ingest_kb.py         # antonia-ingest
    └── benchmark.py         # antonia-bench
```

### Backend Protocol Pattern

Every subsystem exposes a `typing.Protocol`. `AntoniaOrchestrator` depends only on protocols. `factory.py` instantiates the right concrete backend per `AntoniaSettings`. Tests inject `Mock*Backend` objects with zero hardware.

### Prompt System

Prompts live in `config/prompts/*.yaml`. Selected by `settings.prompt_name` (default: `laboratory`). `PromptTemplate.build_messages()` is the only place that assembles the `messages` list sent to the LLM. Adding few-shot examples or changing wording requires only YAML edits, not Python changes.

### Phonetic Map / TTS

`knowledge_base/phonetic_map.yaml` → regex-to-phonetic mappings. Compiled at `load_phonetic_map()` time, not per `speak()`. After `antonia-ingest` updates the YAML, call `preprocessor.load_phonetic_map()` for hot-reload.

---

## Configuration Reference

### Key Profile Fields

| Field | Jetson | Mac M4 | Notes |
|---|---|---|---|
| `stt.compute_type` | `float16` | `float32` | `int8` for ~35-50% more throughput if stable |
| `stt.device` | `cuda` | `cpu` | |
| `llm.keep_alive_seconds` | `0` | `30` | 0 = strict relay; >0 = Qwen stays in VRAM during TTS |
| `llm.num_ctx` | `1024` | `2048` | Jetson memory budget |
| `tts.onnx_providers` | `[CUDA, CPU]` | `[CPU]` | |
| `gpu_relay.enabled` | `true` | `false` | |
| `kb.mode` | `server` | `embedded` | server = Docker ChromaDB |

### Env Var Overrides

```bash
ANTONIA_PROFILE=jetson
ANTONIA_BASE_DIR=/path/to/repo     # auto-detected by default
ANTONIA_MODELS_DIR=/media/...      # models directory
```

---

## Docker / Infrastructure

```bash
make dev-up       # Mac M4: starts Ollama + ChromaDB
make jetson-up    # Jetson: adds host networking + NVIDIA runtime
make down
```

### Jetson Docker (H-4)

`docker/docker-compose.jetson.yml` uses `network_mode: host` to eliminate NAT overhead. Applied as an override: `docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d`.

### Sudoers for `drop_os_cache()` (Jetson only, C-3)

```
mecatronica ALL=(ALL) NOPASSWD: /bin/sh -c sync && echo 3 > /proc/sys/vm/drop_caches
```

---

## Hardware Notes

- **Mic**: USB device index 0 on Jetson; `null` (system default) on Mac M4
- **CTranslate2** on Jetson: built from source at `/media/antonia_ssd/antonia/CTranslate2/python`
- **Whisper compute type**: `float16` is more stable than `int8` on JetPack 6 Tegra
- **After `unload_gpu()`**: Tegra consolidates memory non-deterministically; `TorchMemoryManager` polls up to 1.0s (20 × 50ms)
- **Models directory**: gitignored. Layout: `models/{whisper,kokoro,piper}/`
