from __future__ import annotations

import numpy as np
import numpy.typing as npt
import structlog

log = structlog.get_logger(__name__)


class OpenWakeWordBackend:
    def __init__(self, model_name: str = "hey_jarvis") -> None:
        from openwakeword.model import Model
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        log.info("wakeword_loaded", model=model_name)

    def predict(self, chunk_int16: npt.NDArray[np.int16]) -> dict[str, float]:
        result: dict[str, float] = self._model.predict(chunk_int16)
        return result

    def reset(self) -> None:
        pass
