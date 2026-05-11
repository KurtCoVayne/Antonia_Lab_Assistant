from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class WakeWordBackend(Protocol):
    def predict(self, chunk_int16: npt.NDArray[np.int16]) -> dict[str, float]: ...

    def reset(self) -> None: ...
