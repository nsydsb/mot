from __future__ import annotations

import logging

from logging_utils.ring_buffer import RingBufferHandler

_ring_handler: RingBufferHandler | None = None


def setup_logging(level: str = "INFO", ring_buffer_size: int = 500) -> RingBufferHandler:
    global _ring_handler
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RingBufferHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)
    if _ring_handler is None:
        _ring_handler = RingBufferHandler(ring_buffer_size)
        _ring_handler.setFormatter(fmt)
        root.addHandler(_ring_handler)
    return _ring_handler


def get_ring_handler() -> RingBufferHandler:
    if _ring_handler is None:
        return setup_logging()
    return _ring_handler
