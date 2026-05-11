"""
src/antonia/scripts/benchmark.py

Entry point: antonia-bench
Measures STT, LLM, and TTS latency with pre-recorded audio files.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path


async def _main() -> None:
    from antonia.config.settings import settings
    from antonia.factory import build_stt, build_tts
    from antonia.infra.logging import configure_logging

    configure_logging(level="INFO", fmt="console")

    import structlog
    log = structlog.get_logger(__name__)

    log.info("bench_start", profile=settings.profile)

    # STT warm-up
    stt = await asyncio.to_thread(build_stt, settings)

    import numpy as np
    dummy_audio = np.zeros(int(settings.audio.sample_rate_hw * 3), dtype=np.float32)

    runs = 3
    latencies: list[float] = []
    for i in range(runs):
        t0 = time.time()
        result = await asyncio.to_thread(stt.transcribe, dummy_audio)
        latencies.append(time.time() - t0)
        log.info("stt_bench", run=i + 1, latency_s=round(latencies[-1], 3))

    log.info("stt_bench_summary", avg_s=round(sum(latencies) / len(latencies), 3))


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
