"""
src/antonia/stt/mock.py

MockSTTBackend — returns canned responses. Used in unit tests and CI.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from antonia.domain.utterance import TranscriptionResult


class MockSTTBackend:
    def __init__(self, response: str = "¿Dónde están los multímetros?") -> None:
        self._response = response

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        already_16k: bool = False,
    ) -> TranscriptionResult:
        return TranscriptionResult(text=self._response, latency_s=0.01)

    def unload_gpu(self) -> None:
        pass

    def reload_gpu(self) -> None:
        pass
