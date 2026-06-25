"""Cross-thread data contract and shared lifecycle context for the
decoupled pipeline.

The pipeline is split into three stages — Reader, Inferer, Output — that
communicate exclusively through immutable ``RenderJob`` snapshots carried
by ``queue.Queue``s. All mutable pipeline-wide state (stop signal, error
flag, counters) lives on a single :class:`PipelineCtx` so workers never
share mutable state outside the queues.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderJob:
 """Immutable snapshot passed from Inferer to Output.

 Workers must treat this object as read-only — downstream stages do not
 mutate ``image`` or ``tracks`` in place. ``frame_id`` is the physical
 frame index from the source (monotonically increasing across drops);
 ``tracks`` are the ByteTrack outputs for that exact frame.
 """

 frame_id: int
 image: np.ndarray # BGR, shape (H, W,3), dtype uint8
 tracks: np.ndarray # shape (N,8): [x1, y1, x2, y2, tid, conf, cls, age]
 enqueued_at: float # time.time() when Inferer pushed this job

 @property
 def age_ms(self) -> float:
    return (time.time() - self.enqueued_at) *1000.0


@dataclass
class FrameCounters:
 """Per-stage frame counters, written by the worker that owns them.

 Workers update their own counters under their own (small) lock; the
 Output thread can read ``rendered`` and ``dropped`` for stats logging.
 """

 read: int =0 # frames pulled from FFmpeg source
 inferred: int =0 # frames that completed detect + track
 rendered: int =0 # frames that completed render + sink write
 dropped: int =0 # frames skipped because the output queue was full
 read_failures: int =0 # source.read() raised or returned short read


class PipelineCtx:
 """Shared lifecycle state for the multi-stage pipeline.

 Holds the stop event, error sink, and per-stage counters. Workers must
 not hold any of these locks across blocking I/O or inference calls.
 """

 def __init__(self) -> None:
    self._stop_event = threading.Event()
    self._state_lock = threading.Lock()
    self._error: Optional[str] = None
    self.counters = FrameCounters()
    # One lock per counter field — the counters are independent so
    # contention is negligible but the granularity keeps any single
    # counter update atomic and cheap.
    self._counters_lock = threading.Lock()

 @property
 def stop_event(self) -> threading.Event:
    return self._stop_event

 def request_stop(self) -> None:
    self._stop_event.set()

 @property
 def should_stop(self) -> bool:
    return self._stop_event.is_set()

 def report_error(self, where: str, exc: BaseException) -> None:
    msg = f"{where}: {type(exc).__name__}: {exc}"
    with self._state_lock:
        if self._error is None:
            self._error = msg
            LOGGER.exception("Pipeline worker failed in %s", where)
    # Always request stop on error — the other workers will see this
    # at their next queue get/put and unwind.
    self._stop_event.set()

 @property
 def error(self) -> Optional[str]:
    with self._state_lock:
        return self._error

 # Counter helpers. All four counters share one lock; if profiling
 # later shows contention, split into per-counter locks.
 def incr(self, field_name: str, delta: int =1) -> None:
    with self._counters_lock:
        setattr(self.counters, field_name, getattr(self.counters, field_name) + delta)

 def snapshot(self) -> dict[str, int]:
    with self._counters_lock:
        return {
        "read": self.counters.read,
        "inferred": self.counters.inferred,
        "rendered": self.counters.rendered,
        "dropped": self.counters.dropped,
        "read_failures": self.counters.read_failures,
        }
