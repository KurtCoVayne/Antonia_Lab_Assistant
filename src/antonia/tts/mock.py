from __future__ import annotations

from typing import Optional

import numpy as np

from antonia.domain.utterance import SynthesisResult


class MockTTSBackend:
    def speak(
        self,
        text: str,
        play_audio: bool = True,
        save_wav: bool = False,
        wav_filename: str = "out.wav",
    ) -> Optional[SynthesisResult]:
        samples = np.zeros(16000, dtype=np.float32)
        return SynthesisResult(samples=samples, sample_rate=16000, latency_s=0.01, engine="mock")

    def speak_sentence(self, text: str, force_cpu: bool = False) -> Optional[SynthesisResult]:
        samples = np.zeros(16000, dtype=np.float32)
        return SynthesisResult(samples=samples, sample_rate=16000, latency_s=0.01, engine="mock")
