"""
src/antonia/audio/dsp.py

Pure DSP functions — no model dependencies.
All functions operate on numpy float32 arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def resample(
    audio: npt.NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
    gain: float = 1.0,
) -> npt.NDArray[np.float32]:
    """
    Resample using torchaudio NEON-vectorized sinc_interp_hann.
    Falls back to scipy.signal.resample_poly if torchaudio is unavailable.
    """
    boosted = np.clip(audio * gain, -1.0, 1.0)
    if orig_sr == target_sr:
        return boosted.astype(np.float32)
    try:
        import torch
        import torchaudio

        _get_resampler(orig_sr, target_sr)
        t = torch.from_numpy(boosted).float().unsqueeze(0)
        out = _get_resampler(orig_sr, target_sr)(t).squeeze(0).numpy()
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(target_sr, orig_sr)
        out = resample_poly(boosted, target_sr // g, orig_sr // g).astype(np.float32)
    return out.astype(np.float32)


def preemphasis(audio: npt.NDArray[np.float32], coef: float = 0.97) -> npt.NDArray[np.float32]:
    """High-frequency emphasis for fricatives and stop consonants."""
    return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)


def peak_normalize(
    audio: npt.NDArray[np.float32],
    target: float = 0.95,
) -> npt.NDArray[np.float32]:
    peak = np.max(np.abs(audio))
    if peak > 0:
        return (audio / peak * target).astype(np.float32)
    return audio


def prepare_for_whisper(
    raw: npt.NDArray[np.float32],
    sr_hw: int,
    gain: float = 3.0,
) -> npt.NDArray[np.float32]:
    audio = resample(raw, sr_hw, 16000, gain=gain)
    audio = preemphasis(audio)
    return peak_normalize(audio)


# ── Resampler cache ────────────────────────────────────────────────────────

_resampler_cache: dict[tuple[int, int], object] = {}


def _get_resampler(orig_sr: int, target_sr: int) -> object:
    key = (orig_sr, target_sr)
    if key not in _resampler_cache:
        import torchaudio
        _resampler_cache[key] = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
            resampling_method="sinc_interp_hann",
        )
    return _resampler_cache[key]
