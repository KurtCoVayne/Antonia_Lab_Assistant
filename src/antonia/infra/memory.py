"""
src/antonia/infra/memory.py

TorchMemoryManager — consolidates ad-hoc gc/cache calls from the old pipeline.
Used only when gpu_relay.enabled=True (Jetson); no-ops on other profiles.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import time

import structlog

log = structlog.get_logger(__name__)


class TorchMemoryManager:
    def __init__(
        self,
        target_mb: float = 50.0,
        poll_interval_s: float = 0.05,
        max_polls: int = 20,
    ) -> None:
        self._target_mb = target_mb
        self._poll_interval = poll_interval_s
        self._max_polls = max_polls

    def release_gpu(self) -> None:
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for _ in range(self._max_polls):
                reserved = torch.cuda.memory_reserved() / 1024**2
                if reserved < self._target_mb:
                    break
                torch.cuda.empty_cache()
                time.sleep(self._poll_interval)
            final = torch.cuda.memory_allocated() / 1024**2
            indicator = "ok" if final < self._target_mb else "high"
            log.info("gpu_released", vram_mb=round(final, 1), status=indicator)
        except ImportError:
            pass

    async def drop_os_cache(self) -> None:
        def _sync() -> None:
            try:
                subprocess.run(
                    ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
                    check=True,
                    capture_output=True,
                    timeout=4,
                )
                log.debug("os_cache_dropped")
            except subprocess.TimeoutExpired:
                log.warning("os_cache_drop_timeout")
            except subprocess.CalledProcessError as e:
                log.warning("os_cache_drop_error", returncode=e.returncode)
            except (PermissionError, OSError) as e:
                log.warning("os_cache_drop_permission", error=str(e))

        try:
            await asyncio.wait_for(asyncio.to_thread(_sync), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("os_cache_drop_event_loop_timeout")


class NullMemoryManager(TorchMemoryManager):
    """No-op implementation for non-GPU profiles."""

    def release_gpu(self) -> None:
        pass

    async def drop_os_cache(self) -> None:
        pass
