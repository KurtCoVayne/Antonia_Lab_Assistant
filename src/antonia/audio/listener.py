"""
src/antonia/audio/listener.py

SmartListener — VAD-triggered audio capture. Replaces fixed-duration record_blocking().
Accumulates audio while speech is detected, stops after sustained silence.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from antonia.audio.capture import AudioCapture
from antonia.audio.dsp import resample
from antonia.audio.vad import SileroVAD, VadBuffer


class SmartListener:
    def __init__(
        self,
        capture: AudioCapture,
        vad: SileroVAD,
        sample_rate_hw: int,
        vad_threshold: float = 0.40,
        silence_windows: int = 20,
        min_speech_windows: int = 3,
        max_seconds: float = 15.0,
    ) -> None:
        self._capture = capture
        self._vad = vad
        self._sr_hw = sample_rate_hw
        self._threshold = vad_threshold
        self._silence_windows = silence_windows
        self._min_speech = min_speech_windows
        self._max_seconds = max_seconds

    def listen_until_silence(self) -> npt.NDArray[np.float32]:
        """
        Blocking. Call via asyncio.to_thread().
        Returns raw audio at hardware sample rate when silence is detected.
        """
        self._vad.reset()
        buf = VadBuffer()
        accumulated: list[npt.NDArray[np.float32]] = []
        speech_windows = 0
        silence_streak = 0
        deadline = time.monotonic() + self._max_seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            chunk = self._capture.get_chunk_timeout(min(remaining, 1.0))
            if chunk is None:
                if time.monotonic() >= deadline:
                    break
                continue

            accumulated.append(chunk)

            resampled = resample(chunk, self._sr_hw, 16_000)
            buf.push(resampled)
            prob = self._vad.score_buffer(buf)

            if prob >= self._threshold:
                speech_windows += 1
                silence_streak = 0
            elif speech_windows >= self._min_speech:
                silence_streak += 1

            if speech_windows >= self._min_speech and silence_streak >= self._silence_windows:
                break

        if not accumulated:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(accumulated)
