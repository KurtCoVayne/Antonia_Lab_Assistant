"""
scripts/download_models.py

Downloads all model files required by Antonia on Mac (apple-silicon profile):
  - Kokoro ONNX v1.0  →  models/kokoro/
  - Piper voices      →  models/piper/
  - Whisper small     →  ~/.cache/huggingface (via faster-whisper auto-download)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-whisper
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1].resolve()
MODELS_DIR = REPO_ROOT / "models"

# ── File registry ──────────────────────────────────────────────────────────────

_KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

KOKORO_FILES = [
    (
        f"{_KOKORO_BASE}/kokoro-v1.0.onnx",
        MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx",
        None,
    ),
    (
        f"{_KOKORO_BASE}/voices-v1.0.bin",
        MODELS_DIR / "kokoro" / "voices-v1.0.bin",
        None,
    ),
]

# Piper voice files: model + JSON config per voice
PIPER_FILES = [
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx",
        MODELS_DIR / "piper" / "es_MX-claude-high.onnx",
        None,
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
        MODELS_DIR / "piper" / "es_MX-claude-high.onnx.json",
        None,
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        MODELS_DIR / "piper" / "en_US-lessac-medium.onnx",
        None,
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        MODELS_DIR / "piper" / "en_US-lessac-medium.onnx.json",
        None,
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.1f} MB"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Progress:
    def __init__(self, filename: str) -> None:
        self._name = filename
        self._last = -1

    def __call__(self, count: int, block: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(count * block * 100 / total))
        if pct != self._last and pct % 5 == 0:
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {self._name}", end="", flush=True)
            self._last = pct
        if pct == 100:
            print()


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_sha256 and _sha256(dest) != expected_sha256:
            print(f"  [warn] checksum mismatch, re-downloading {dest.name}")
        else:
            size = dest.stat().st_size
            print(f"  [skip] {dest.name} already exists ({_fmt_mb(size)})")
            return

    print(f"  [down] {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        # Use curl for reliable redirect following (GitHub releases use multi-hop redirects)
        result = subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", str(tmp), url],
            check=True,
        )
        if expected_sha256 and _sha256(tmp) != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {dest.name}")
        tmp.rename(dest)
        print(f"  [ok]   {dest.name} ({_fmt_mb(dest.stat().st_size)})")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_whisper(model_size: str = "small") -> None:
    """
    faster-whisper auto-downloads on first model load; we trigger it here
    so startup is instant later.
    """
    print(f"\n[Whisper] Downloading '{model_size}' via faster-whisper …")
    try:
        from faster_whisper import WhisperModel
        WhisperModel(model_size, device="cpu", compute_type="float32")
        print(f"  [ok]   Whisper '{model_size}' ready")
    except ImportError:
        print("  [warn] faster-whisper not installed — skipping Whisper download")
    except Exception as exc:
        print(f"  [fail] {exc}", file=sys.stderr)
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Antonia model files")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip Whisper download")
    parser.add_argument("--skip-piper", action="store_true", help="Skip Piper voices download")
    parser.add_argument("--skip-kokoro", action="store_true", help="Skip Kokoro download")
    args = parser.parse_args()

    print(f"Models directory: {MODELS_DIR}\n")

    if not args.skip_kokoro:
        print("[Kokoro] Downloading kokoro-v1.0 ONNX …")
        for url, dest, sha in KOKORO_FILES:
            download_file(url, dest, sha)

    if not args.skip_piper:
        print("\n[Piper] Downloading voices …")
        for url, dest, sha in PIPER_FILES:
            download_file(url, dest, sha)

    if not args.skip_whisper:
        download_whisper("small")

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
