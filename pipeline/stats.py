from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatsSnapshot:
    """A point-in-time view of the in-scene track population.

    `counts` and `tracked_ids` share the same keyset (the configured category
    names). `total` is the sum of all category counts. `window_sec` echoes the
    active-window configuration so callers can interpret the values without
    having to re-fetch the config.
    """

    counts: dict[str, int]
    tracked_ids: dict[str, list[int]]
    total: int
    window_sec: float
    taken_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "window_sec": self.window_sec,
            "categories": self.counts,
            "tracked_ids": self.tracked_ids,
            "taken_at": self.taken_at,
        }


class TrackCategoryCounter:
    """Count unique track ids per named category over a sliding time window.

    The counter keeps, for every observed `tid`, the last frame's class id and
    the monotonic timestamp when it was last seen. A snapshot considers a tid
    "active" iff `now - last_seen <= active_window_sec`. This makes the count
    robust to brief occlusions or off-frame glitches that would otherwise cause
    the population to flicker.
    """

    def __init__(
        self,
        category_map: Mapping[str, Sequence[int]],
        active_window_sec: float = 5.0,
    ) -> None:
        if active_window_sec <= 0:
            raise ValueError("active_window_sec must be > 0")
        # Normalize: dict[str, frozenset[int]] for O(1) lookups.
        self._category_map: dict[str, frozenset[int]] = {
            name: frozenset(int(c) for c in classes) for name, classes in category_map.items()
        }
        self._active_window_sec = float(active_window_sec)
        self._tid_state: dict[int, tuple[int, float]] = {}
        self._lock = threading.Lock()

    @property
    def categories(self) -> list[str]:
        return list(self._category_map.keys())

    @property
    def active_window_sec(self) -> float:
        return self._active_window_sec

    def update(self, tracks: np.ndarray, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        if tracks is None or tracks.size == 0:
            self._evict(now)
            return
        # Track rows are [x1, y1, x2, y2, tid, conf, cls, age].
        with self._lock:
            for trk in tracks:
                if len(trk) < 7:
                    continue
                tid = int(trk[4])
                cls = int(trk[6])
                self._tid_state[tid] = (cls, now)
            self._evict_locked(now)

    def snapshot(self, now: float | None = None) -> StatsSnapshot:
        if now is None:
            now = time.time()
        with self._lock:
            self._evict_locked(now)
            # Build reverse map: class id -> category name.
            cls_to_cat: dict[int, str] = {}
            for cat, cls_set in self._category_map.items():
                for cid in cls_set:
                    cls_to_cat[cid] = cat
            cutoff = now - self._active_window_sec
            per_cat_ids: dict[str, list[int]] = {cat: [] for cat in self._category_map}
            counts: dict[str, int] = {cat: 0 for cat in self._category_map}
            for tid, (cls, last_seen) in self._tid_state.items():
                if last_seen < cutoff:
                    continue
                cat = cls_to_cat.get(cls)
                if cat is None:
                    continue
                counts[cat] += 1
                per_cat_ids[cat].append(tid)
            for cat_ids in per_cat_ids.values():
                cat_ids.sort()
            total = sum(counts.values())
            return StatsSnapshot(
                counts=counts,
                tracked_ids=per_cat_ids,
                total=total,
                window_sec=self._active_window_sec,
                taken_at=now,
            )

    def _evict(self, now: float) -> None:
        with self._lock:
            self._evict_locked(now)

    def _evict_locked(self, now: float) -> None:
        cutoff = now - self._active_window_sec
        stale = [tid for tid, (_, last_seen) in self._tid_state.items() if last_seen < cutoff]
        for tid in stale:
            self._tid_state.pop(tid, None)


class StatsReporter:
    """Periodically log a stats snapshot from a :class:`TrackCategoryCounter`.

    The reporter runs on a daemon thread; the main pipeline thread only
    forwards per-frame updates to the counter, so the periodic logging has no
    impact on inference latency.
    """

    def __init__(
        self,
        counter: TrackCategoryCounter,
        interval_sec: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        self._counter = counter
        self._interval_sec = float(interval_sec)
        self._logger = logger or LOGGER
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interval_sec(self) -> float:
        return self._interval_sec

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="mot-stats-reporter", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        # Emit one report immediately so the operator gets fast feedback on
        # startup, then settle into the periodic cadence.
        self._emit("initial")
        while not self._stop_event.wait(self._interval_sec):
            self._emit("periodic")

    def _emit(self, reason: str) -> None:
        try:
            snap = self._counter.snapshot()
        except Exception:
            self._logger.exception("Failed to take stats snapshot (%s)", reason)
            return
        self._logger.info(
            "scene stats (%s, window=%.1fs): %s | total=%d",
            reason,
            snap.window_sec,
            snap.counts,
            snap.total,
        )
