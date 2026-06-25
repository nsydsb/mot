from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

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
        # Pub/sub plumbing. Each subscriber gets its own bounded queue so a
        # slow consumer cannot stall the inferer thread that calls
        # ``publish``. ``_subscribers_lock`` guards the dict; each subscriber
        # owns its own queue and is responsible for draining / closing it.
        self._subscribers: dict[int, "queue.Queue[StatsSnapshot]"] = {}
        self._subscribers_lock = threading.Lock()
        self._next_subscriber_id = 0

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

    def subscribe(self, queue_size: int = 8) -> tuple[int, "queue.Queue[StatsSnapshot]"]:
        """Register a new snapshot subscriber.

        Returns ``(subscriber_id, queue)``. The caller drains the queue;
        when it's done it MUST call :meth:`unsubscribe` with the id to
        release resources. ``queue_size`` bounds how many snapshots can
        queue up before older ones are dropped (``put_nowait`` is used
        by the publisher), so a slow consumer cannot stall the inferer.
        """
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        with self._subscribers_lock:
            sub_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[sub_id] = queue.Queue(maxsize=queue_size)
        return sub_id, self._subscribers[sub_id]

    def unsubscribe(self, sub_id: int) -> None:
        """Remove a previously registered subscriber."""
        with self._subscribers_lock:
            self._subscribers.pop(sub_id, None)

    def close_all_subscribers(self) -> None:
        """Tear down every subscriber queue and signal consumers to exit.

        Called by the pipeline on stop. Each queue gets a ``None``
        sentinel pushed via ``put_nowait``; the WebSocket bridge
        interprets ``None`` as "pipeline is shutting down, close
        cleanly". We also drop the dict so any late ``subscribe`` /
        ``unsubscribe`` calls become no-ops.
        """
        with self._subscribers_lock:
            queues = list(self._subscribers.values())
            self._subscribers.clear()
        for q in queues:
            try:
                q.put_nowait(None)  # type: ignore[arg-type]
            except queue.Full:
                # Drop a slot so the sentinel fits — better to overwrite
                # an old snapshot than to leave the consumer hanging.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)  # type: ignore[arg-type]
                except queue.Full:
                    pass

    def publish(self, snap: StatsSnapshot) -> None:
        """Fan a snapshot out to every registered subscriber.

        Called from the publisher thread (NOT from ``update`` — we don't
        want a snapshot per inference frame; the publisher throttles to
        the configured interval). Uses ``put_nowait`` so a stuck
        subscriber simply drops the new snapshot instead of blocking
        everyone else.
        """
        # Snapshot the subscriber list under the lock, then release it
        # before doing queue puts — keeps the critical section short and
        # avoids holding the lock while touching per-subscriber queues.
        with self._subscribers_lock:
            queues = list(self._subscribers.values())
        for q in queues:
            try:
                q.put_nowait(snap)
            except queue.Full:
                # Slow consumer: drop the snapshot for them. They'll
                # catch up on the next tick.
                LOGGER.debug("stats subscriber queue full; dropping snapshot")

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


class StatsPublisher:
    """Push :class:`StatsSnapshot` to subscribers at a fixed cadence.

    A daemon thread wakes every ``interval_sec``, takes one snapshot from
    the counter, and fans it out to every registered subscriber via
    :meth:`TrackCategoryCounter.publish`. Decoupling the snapshot
    frequency from the inference rate (which can be 30+ fps) keeps the
    cost of each ``snapshot()`` — O(active tracks) — bounded, and lets
    WebSocket clients receive a steady, predictable stream.

    The publisher is a no-op when constructed with a ``None`` counter,
    so the pipeline can wire it unconditionally.
    """

    def __init__(
        self,
        counter: TrackCategoryCounter | None,
        interval_sec: float = 1.0,
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
        if self._counter is None:
            # Nothing to publish; don't even spawn a thread.
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="mot-stats-publisher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Wake up any WebSocket handlers so they can close the
        # connection promptly instead of waiting for their next
        # ``get(timeout=...)`` to time out.
        if self._counter is not None:
            self._counter.close_all_subscribers()

    def _run(self) -> None:
        # Emit one immediately so a freshly-connected WebSocket gets data
        # without waiting a full interval, then settle into cadence.
        self._emit("initial")
        while not self._stop_event.wait(self._interval_sec):
            self._emit("periodic")

    def _emit(self, reason: str) -> None:
        try:
            snap = self._counter.snapshot()  # type: ignore[union-attr]
        except Exception:
            self._logger.exception("Failed to take stats snapshot (%s)", reason)
            return
        try:
            self._counter.publish(snap)  # type: ignore[union-attr]
        except Exception:
            self._logger.exception("Failed to publish stats snapshot (%s)", reason)
