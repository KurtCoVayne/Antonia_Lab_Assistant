"""
src/antonia/stt/backend.py

STTBackend protocol — structural typing, no ABC inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from antonia.domain.utterance import TranscriptionResult


@runtime_checkable
class STTBackend(Protocol):
    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        already_16k: bool = False,
    ) -> TranscriptionResult: ...

    def unload_gpu(self) -> None: ...

    def reload_gpu(self) -> None: ...
