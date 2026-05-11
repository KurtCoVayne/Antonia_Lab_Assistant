"""
src/antonia/audio/vad.py

Silero VAD wrapper with pre-allocated circular buffer.
The VadBuffer eliminates heap allocations in the 10Hz audio callback hot path.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

VAD_WINDOW = 512   # Silero requires exactly 512 samples at 16kHz
_BUF_SIZE  = 3000  # 2 × VAD_WINDOW + 1604-sample chunk headroom


class VadBuffer:
    """
    Pre-allocated buffer for VAD sample accumulation.
    push() is a single memcpy; score() processes 512-sample windows in-place.
    """

    def __init__(self, capacity: int = _BUF_SIZE) -> None:
        self._buf: npt.NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._len = 0

    def reset(self) -> None:
        self._len = 0

    def push(self, chunk: npt.NDArray[np.float32]) -> None:
        n = len(chunk)
        end = self._len + n
        if end > len(self._buf):
            new_buf = np.zeros(end * 2, dtype=np.float32)
            new_buf[: self._len] = self._buf[: self._len]
            self._buf = new_buf
        self._buf[self._len : end] = chunk
        self._len = end

    def score(self, vad_model: object, sample_rate: int = 16000) -> float:
        """
        Process all complete 512-sample windows. Returns max VAD probability.
        Residual samples (< 512) are compacted to the front for the next push().
        """
        import torch

        max_prob = 0.0
        offset = 0

        while offset + VAD_WINDOW <= self._len:
            window = self._buf[offset : offset + VAD_WINDOW]
            t = torch.from_numpy(window.copy())
            prob = vad_model(t, sample_rate).item()  # type: ignore[operator]
            max_prob = max(max_prob, prob)
            offset += VAD_WINDOW

        remainder = self._len - offset
        if remainder > 0:
            self._buf[:remainder] = self._buf[offset : self._len]
        self._len = remainder
        return max_prob


class SileroVAD:
    """Thin wrapper around the silero_vad model."""

    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._model.reset_states()

    def reset(self) -> None:
        self._model.reset_states()

    def score_buffer(self, buf: VadBuffer, sample_rate: int = 16000) -> float:
        return buf.score(self._model, sample_rate)
