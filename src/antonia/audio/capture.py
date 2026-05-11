"""
src/antonia/audio/capture.py

AudioCapture — sounddevice abstraction.
The callback enqueues chunks without processing; all DSP happens on the consumer thread.
"""

from __future__ import annotations

import queue
from typing import Any

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import structlog

log = structlog.get_logger(__name__)


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_samples: int = 4410,
        device: int | None = None,
        maxsize: int = 80,
    ) -> None:
        self._sr = sample_rate
        self._chunk = chunk_samples
        self._device = device
        self._queue: queue.Queue[npt.NDArray[np.float32]] = queue.Queue(maxsize=maxsize)
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=self._chunk,
            device=self._device,
            latency="high",
            callback=self._callback,
        )
        self._stream.__enter__()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.__exit__(None, None, None)
            self._stream = None

    def get_chunk(self) -> npt.NDArray[np.float32]:
        return self._queue.get()

    def _callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        if status.input_overflow:
            return
        if status:
            log.warning("audio_callback_status", status=str(status))
        try:
            self._queue.put_nowait(indata.flatten().copy())
        except queue.Full:
            pass

    def record_blocking(self, seconds: float) -> npt.NDArray[np.float32]:
        """Synchronous recording — for pipeline test scripts."""
        audio = sd.rec(
            int(seconds * self._sr),
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            device=self._device,
        )
        sd.wait()
        return audio.flatten().astype(np.float32)
