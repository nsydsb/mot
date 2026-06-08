from __future__ import annotations

import logging
import threading
import time
import uuid

from config.schema import AppConfig
from detectors.yolo_detector import YoloDetector
from pipeline.result import PipelineState, PipelineStatus
from pipeline.stats import StatsReporter, StatsSnapshot, TrackCategoryCounter
from render.renderer import Renderer
from sink.ffmpeg_srs_sink import FFmpegSrsSink
from source.ffmpeg_source import FFmpegSource
from trackers.adapter import ByteTrackAdapter

LOGGER = logging.getLogger(__name__)


class TrackingPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.task_id = uuid.uuid4().hex
        self.status = PipelineStatus.IDLE
        self.error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.source: FFmpegSource | None = None
        self.detector: YoloDetector | None = None
        self.tracker: ByteTrackAdapter | None = None
        self.renderer: Renderer | None = None
        self.sink: FFmpegSrsSink | None = None
        # Stats components are constructed up-front so the manager can answer
        # stats queries via snapshot() even before the inference thread has
        # seen its first frame.
        self.stats_counter: TrackCategoryCounter | None = None
        self.stats_reporter: StatsReporter | None = None
        if cfg.stats.enabled:
            self.stats_counter = TrackCategoryCounter(
                cfg.stats.category_map, cfg.stats.active_window_sec
            )
            self.stats_reporter = StatsReporter(
                self.stats_counter,
                cfg.stats.report_interval_sec,
                logger=logging.getLogger("mot.stats"),
            )

    def start(self) -> PipelineState:
        with self._lock:
            if self.status in {PipelineStatus.STARTING, PipelineStatus.RUNNING}:
                raise RuntimeError("Pipeline is already running")
            self.status = PipelineStatus.STARTING
            self.error = None
            self._stop_event.clear()
            self.sink = FFmpegSrsSink(self.cfg.sink)
            if self.stats_reporter is not None:
                self.stats_reporter.start()
            self._thread = threading.Thread(target=self._run, name=f"mot-{self.task_id}", daemon=True)
            self._thread.start()
            return self.state()

    def stop(self) -> PipelineState:
        with self._lock:
            if self.status not in {PipelineStatus.STARTING, PipelineStatus.RUNNING, PipelineStatus.ERROR}:
                return self.state()
            self.status = PipelineStatus.STOPPING
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self.state()

    def state(self) -> PipelineState:
        sink = self.sink or FFmpegSrsSink(self.cfg.sink)
        return PipelineState(
            task_id=self.task_id,
            status=self.status,
            stream_name=self.cfg.sink.stream_name,
            play_urls=sink.play_urls,
            error=self.error,
        )

    def stats_snapshot(self) -> StatsSnapshot | None:
        """Return the current in-scene population snapshot, or None if stats
        are disabled / no pipeline run is in progress."""
        if self.stats_counter is None:
            return None
        return self.stats_counter.snapshot()

    def _run(self) -> None:
        try:
            self.source = FFmpegSource(self.cfg.source)
            self.detector = YoloDetector(self.cfg.detector)
            self.tracker = ByteTrackAdapter(self.cfg.tracker)
            self.renderer = Renderer(self.cfg.render)
            self.sink = self.sink or FFmpegSrsSink(self.cfg.sink)
            self.source.start()
            self.sink.start()
            self.status = PipelineStatus.RUNNING
            LOGGER.info("Pipeline started task_id=%s stream=%s", self.task_id, self.cfg.sink.stream_name)
            while not self._stop_event.is_set():
                try:
                    pkt = self.source.read()
                except Exception:
                    LOGGER.exception("Source read failed; retrying")
                    time.sleep(self.cfg.source.reconnect_delay_sec)
                    continue
                dets = self.detector.predict(pkt.image)
                tracks = self.tracker.update(dets, pkt.image)
                if self.stats_counter is not None:
                    self.stats_counter.update(tracks)
                rendered = self.renderer.render(pkt.image, tracks)
                self.sink.write(rendered)
        except Exception as exc:
            self.error = str(exc)
            self.status = PipelineStatus.ERROR
            LOGGER.exception("Pipeline failed")
        finally:
            self._release()
            if self.status == PipelineStatus.STOPPING:
                self.status = PipelineStatus.STOPPED
            LOGGER.info("Pipeline finished task_id=%s status=%s", self.task_id, self.status)

    def _release(self) -> None:
        if self.stats_reporter is not None:
            try:
                self.stats_reporter.stop()
            except Exception:
                LOGGER.exception("Failed to stop stats reporter")
        for component in (self.source, self.sink):
            if component is None:
                continue
            try:
                component.stop()
            except Exception:
                LOGGER.exception("Failed to stop component %s", type(component).__name__)
