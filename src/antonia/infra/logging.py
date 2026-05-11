"""
src/antonia/infra/logging.py

structlog configuration. Call configure_logging() once at application startup
before any other module initializes. All modules acquire loggers via:
    log = structlog.get_logger(__name__)
"""

from __future__ import annotations

import logging
from typing import Literal

import structlog


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    fmt: Literal["console", "json"] = "console",
) -> None:
    stdlib_level = getattr(logging, level, logging.INFO)
    logging.basicConfig(level=stdlib_level, format="%(message)s")

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if fmt == "json":
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        wrapper_class = structlog.make_filtering_bound_logger(stdlib_level)
        structlog.configure(
            processors=processors,
            wrapper_class=wrapper_class,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        processors_console: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        wrapper_class = structlog.make_filtering_bound_logger(stdlib_level)
        structlog.configure(
            processors=processors_console,
            wrapper_class=wrapper_class,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
