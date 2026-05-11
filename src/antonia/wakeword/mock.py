from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MockWakeWordBackend:
    """Always returns a score of 0 (never triggers)."""

    def predict(self, chunk_int16: npt.NDArray[np.int16]) -> dict[str, float]:
        return {"hey_jarvis": 0.0}

    def reset(self) -> None:
        pass
