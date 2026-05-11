"""
tests/unit/test_dsp.py

Unit tests for audio DSP functions — no hardware, no models.
"""

import numpy as np
import pytest

from antonia.audio.dsp import peak_normalize, preemphasis, prepare_for_whisper


def _sine(freq: float = 440.0, sr: int = 44100, duration: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def test_preemphasis_length_preserved():
    audio = _sine()
    out = preemphasis(audio)
    assert len(out) == len(audio)


def test_preemphasis_dtype():
    audio = _sine()
    out = preemphasis(audio)
    assert out.dtype == np.float32


def test_peak_normalize_peak_is_target():
    audio = _sine()
    out = peak_normalize(audio, target=0.95)
    assert abs(np.max(np.abs(out)) - 0.95) < 1e-5


def test_peak_normalize_silent():
    audio = np.zeros(1000, dtype=np.float32)
    out = peak_normalize(audio)
    assert np.all(out == 0.0)


def test_prepare_for_whisper_output_length():
    audio = _sine(sr=44100, duration=1.0)
    out = prepare_for_whisper(audio, sr_hw=44100, gain=1.0)
    # Resampled from 44100 to 16000 Hz → ~16000 samples for 1s
    assert 15000 < len(out) < 17000


def test_prepare_for_whisper_peak_bounded():
    audio = _sine() * 5.0  # deliberately loud
    out = prepare_for_whisper(audio, sr_hw=44100, gain=1.0)
    assert np.max(np.abs(out)) <= 1.0
