from __future__ import annotations

import copy
import logging
from threading import Lock
from typing import Any

from config.schema import AppConfig
from pipeline.pipeline import TrackingPipeline
from pipeline.result import PipelineState, PipelineStatus

LOGGER = logging.getLogger(__name__)

ALLOWED_API_OVERRIDES = {
    "source.fps",
    "source.width",
    "source.height",
    "detector.model_type",
    "detector.conf",
    "detector.iou",
    "detector.imgsz",
    "detector.classes",
    "tracker.track_thresh",
    "tracker.low_thresh",
    "tracker.match_thresh",
}


class DetectionManager:
    def __init__(self, base_config: AppConfig):
        self.base_config = base_config
        self.pipeline: TrackingPipeline | None = None
        self.lock = Lock()

    def start(
        self,
        source_url: str | None,
        stream_name: str | None,
        model_type: str | None,
        classes: list[int | str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineState:
        with self.lock:
            if self.pipeline and self.pipeline.status in {PipelineStatus.STARTING, PipelineStatus.RUNNING}:
                raise RuntimeError("A detection task is already running")
            merged_overrides = dict(overrides or {})
            if classes is not None:
                # Explicit classes field takes precedence over the same key
                # coming in through overrides.
                merged_overrides["detector.classes"] = classes
            clean_overrides = self._validate_overrides(merged_overrides)
            data = copy.deepcopy(self.base_config.model_dump())
            if source_url:
                data["source"]["url"] = source_url
            if stream_name:
                data["sink"]["stream_name"] = stream_name
            if model_type:
                data["detector"]["model_type"] = model_type
            cfg = AppConfig.model_validate(data).with_overrides(clean_overrides)
            cfg.sink.width = cfg.source.width
            cfg.sink.height = cfg.source.height
            cfg.sink.fps = cfg.source.fps
            self.pipeline = TrackingPipeline(cfg)
            return self.pipeline.start()

    def stop(self) -> PipelineState:
        with self.lock:
            if not self.pipeline:
                return PipelineState(status=PipelineStatus.IDLE)
            pipeline = self.pipeline
        state = pipeline.stop()
        # Clear the reference so new WebSocket subscribers can't latch
        # onto a stopped pipeline (which would never publish again).
        # Done outside the manager lock to avoid a deadlock with the
        # pipeline's own internal locks.
        with self.lock:
            if self.pipeline is pipeline:
                self.pipeline = None
        return state

    def stats(self) -> dict[str, Any] | None:
        """Return a JSON-serializable stats view, or None if no pipeline is
        running or stats are disabled in the config."""
        with self.lock:
            pipeline = self.pipeline
        if pipeline is None:
            return None
        snap = pipeline.stats_snapshot()
        if snap is None:
            return None
        return {
            "task_id": pipeline.task_id,
            "status": pipeline.status.value,
            "stream_name": pipeline.cfg.sink.stream_name,
            "report_interval_sec": self.base_config.stats.report_interval_sec,
            "active_window_sec": snap.window_sec,
            "total": snap.total,
            "categories": snap.counts,
            "tracked_ids": snap.tracked_ids,
            "taken_at": snap.taken_at,
        }

    def current_pipeline(self) -> "TrackingPipeline | None":
        """Return the active pipeline, or None if none is running.

        Public accessor used by long-lived endpoints (e.g. the stats
        WebSocket) that need to attach to a pipeline without racing the
        ``stop()`` call.
        """
        with self.lock:
            return self.pipeline

    @staticmethod
    def _validate_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
        for key in overrides:
            if key not in ALLOWED_API_OVERRIDES:
                raise KeyError(f"Override is not allowed: {key}")
        return overrides
