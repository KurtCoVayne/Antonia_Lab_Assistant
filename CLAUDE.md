# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Antonia is a voice-controlled AI assistant for the Digital Control Laboratory at Universidad EAFIT (Medellín, Colombia). It runs on a **NVIDIA Jetson Orin Nano 8GB** with a USB microphone. The pipeline is: Wake Word → STT → LLM → TTS.

All code lives in `antonia_project/`. Models are stored externally on `/media/antonia_ssd/antonia/antonia_project/models/`.

## Running the Pipeline

```bash
# Full STT→LLM→TTS pipeline test (runs N_TURNOS cycles with manual recording)
cd antonia_project
python llm/test_stt_llm_tts.py

# Wake word + VAD + STT only (continuous loop, requires "Hey Jarvis")
python modules/stt/antonia_core.py

# Individual module tests
python tests/whisper_test.py
python tests/audio_test.py
python llm/llm_test.py
```

## LLM — Ollama Docker

Qwen 2.5 1.5B runs inside a Docker container (`ollama_antonia`):

```bash
# Start container
sudo docker start ollama_antonia

# Pull model (if not present)
sudo docker exec -it ollama_antonia ollama pull qwen2.5:1.5b

# Create custom Modelfile (sets num_ctx=1024, temperature=0.1, num_predict=160)
sudo docker exec -it ollama_antonia ollama create antonia-llm -f /path/to/config/Modelfile
```

The container is configured with `KEEP_ALIVE=0` so Qwen auto-unloads from GPU after each response. API endpoint: `http://localhost:11434/api/chat`.

## Architecture

### GPU Relay System (Critical Constraint)

The Jetson has **unified memory** (~8 GB shared between CPU and GPU). Whisper and Qwen cannot coexist in VRAM. The relay cycle is:

```
STT: Whisper on GPU (~900 MB)
  → stt.unload_gpu() + drop_os_cache()   ← clears VRAM for Qwen
LLM: Qwen on GPU (~1600 MB)
  → KEEP_ALIVE=0 auto-unloads Qwen
TTS: Kokoro on CPU (0 MB VRAM)
  → stt.reload_gpu()                      ← Whisper returns
```

`AntoniaSTT` (`modules/stt/stt_module.py`) exposes `unload_gpu()` and `reload_gpu()` for this handoff.

### STT Module (`modules/stt/`)

- **`stt_module.py`** — `AntoniaSTT` class: Whisper small, float16, GPU. DSP pipeline: gain → librosa resample 44100→16000 Hz → pre-emphasis → peak normalization.
- **`antonia_core.py`** — Standalone wake-word loop. State machine: `SLEEPING → FLUSHING → LISTENING → (PROCESSING in thread)`. Uses OpenWakeWord (`hey_jarvis`), Silero VAD, and Whisper. Discards 600ms of audio post-wake-word (WW_FLUSH_CHUNKS=6) so Whisper doesn't transcribe "Hey Jarvis".

Key tuning constants in `antonia_core.py`:
- `NOISE_GATE_RMS = 0.008` — adjust for lab background noise (fans, equipment)
- `VAD_THRESHOLD = 0.40`, `SILENCE_WINDOWS = 20` — controls utterance boundary detection
- `LISTENING_TIMEOUT_CHUNKS = 60` — 6-second timeout before returning to sleep

### TTS Module (`modules/tts/tts_module.py`)

Four-layer pipeline:
1. **TextPreprocessor** — strips LLM markdown, expands units (kHz→kilohercios), applies phonetic substitutions
2. **LanguageDetector** — ES/EN detection via `langdetect` or word-frequency heuristic
3. **KokoroEngine** — primary: Kokoro-82M ONNX on CPU (`ef_dora` for ES, `af_bella` for EN)
4. **PiperEngine** — fallback: Piper CLI subprocess

Phonetic map lives in `knowledge_base/phonetic_map.json` (regex keys → phonetic strings). Load hot-reload via `tts.preprocessor.load_phonetic_map()` after updates — no restart needed. Keys starting with `_` are comments and are ignored.

**RAG integration point**: When a RAG ingestion pipeline adds new technical terms to the lab knowledge base, it should also append entries to `phonetic_map.json` and call `load_phonetic_map()`. See `PUERTA_RAG` comment in `tts_module.py`.

Global TTS instance: `from modules.tts.tts_module import tts`.

### LLM Client (`llm/test_stt_llm_tts.py`)

Uses `httpx.AsyncClient` with `keep_alive=0` in the payload to trigger Ollama's auto-unload. Maintains conversation history (max 4 messages = 2 turns). Retries on OOM errors with `drop_os_cache()`.

### System Prompt / Persona

`config/system_prompt.txt` defines Antonia's persona: bilingual (understands ES/EN/Spanglish, always responds in Spanish), plain text only (no markdown — output goes directly to TTS), max 3–4 short sentences, zero hallucinations (RAG-only answers).

## Docker / Networking (H-4)

To eliminate Docker NAT overhead (~1–2ms/request) on the Ollama container, recreate it with `--network host`:

```bash
sudo docker stop ollama_antonia && sudo docker rm ollama_antonia
sudo docker run -d --name ollama_antonia --network host \
  --gpus all -v ollama_data:/root/.ollama dustynv/ollama
```

This collapses the path to a direct loopback connection (no iptables NAT). Acceptable for a single-purpose edge device.

## Sudoers for `drop_os_cache()` (C-3)

The pipeline calls `sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"`. Without NOPASSWD, this blocks indefinitely if the sudo session expires. Add to `/etc/sudoers` on the Jetson:

```
mecatronica ALL=(ALL) NOPASSWD: /bin/sh -c sync && echo 3 > /proc/sys/vm/drop_caches
```

## Key Tuning Constants

| Constant | File | Default | Notes |
|---|---|---|---|
| `COMPUTE_TYPE` | `stt_module.py` | `"float16"` | Switch to `"int8"` for ~35-50% more STT throughput if stable on JetPack 6 |
| `OLLAMA_KEEP_ALIVE_S` | `test_stt_llm_tts.py` | `0` | Increase to ~10s to reduce Qwen cold-load latency (trade: Qwen stays in VRAM during TTS) |
| `NOISE_GATE_RMS` | `antonia_core.py` | `0.008` | Increase if lab has high background noise |
| `SILENCE_WINDOWS` | `antonia_core.py` | `20` | 20 × 32ms = ~640ms silence before end-of-utterance |

## Hardware Notes

- **Mic**: USB device index 0, native 44100 Hz — always resampled to 16000 Hz for Whisper/VAD
- **Whisper compute type**: `float16` (more stable than `int8` on JetPack 6 Tegra)
- **CTranslate2**: installed from local build (`/media/antonia_ssd/antonia/CTranslate2/python`)
- After `unload_gpu()`, wait ~300–500ms for Tegra to consolidate physical memory blocks — `time.sleep(0.5)` is intentional
