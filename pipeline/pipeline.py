from __future__ import annotations

import logging
import queue
import threading
import time
import uuid

from config.schema import AppConfig
from detectors.yolo_detector import YoloDetector
from pipeline.jobs import PipelineCtx
from pipeline.result import PipelineState, PipelineStatus
from pipeline.stats import StatsPublisher, StatsReporter, StatsSnapshot, TrackCategoryCounter
from pipeline.workers import inferer_worker, output_worker, reader_worker
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
        self.ctx: PipelineCtx | None = None
        # Stats components are constructed up-front so the manager can answer
        # stats queries via snapshot() even before the inference thread has
        # seen its first frame.
        self.stats_counter: TrackCategoryCounter | None = None
        self.stats_reporter: StatsReporter | None = None
        self.stats_publisher: StatsPublisher | None = None
        if cfg.stats.enabled:
            self.stats_counter = TrackCategoryCounter(
                cfg.stats.category_map, cfg.stats.active_window_sec
            )
        # StatsReporter only logs; skip it entirely when stats are
        # disabled rather than letting it tick on a None counter.
        if self.stats_counter is not None:
            self.stats_reporter = StatsReporter(
                self.stats_counter,
                cfg.stats.report_interval_sec,
                logger=logging.getLogger("mot.stats"),
            )
            # Publisher pushes 1Hz snapshots to WebSocket subscribers.
            # Cadence is configurable via cfg.stats.publish_interval_sec.
            self.stats_publisher = StatsPublisher(
                self.stats_counter,
                cfg.stats.publish_interval_sec,
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
            if self.stats_publisher is not None:
                self.stats_publisher.start()
            self._thread = threading.Thread(
                target=self._run,
                name=f"mot-{self.task_id}",
                daemon=True,
            )
            self._thread.start()
            return self.state()

    def stop(self) -> PipelineState:
        with self._lock:
            if self.status not in {PipelineStatus.STARTING, PipelineStatus.RUNNING, PipelineStatus.ERROR}:
                return self.state()
            self.status = PipelineStatus.STOPPING
            self._stop_event.set()
            # Also notify the in-pipeline ctx so worker loops can wake up
            # promptly from queue.get(timeout=...).
            if self.ctx is not None:
                self.ctx.request_stop()
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
        # Return the current in-scene population snapshot, or None if stats
        # are disabled / no pipeline run is in progress.
        if self.stats_counter is None:
            return None
        return self.stats_counter.snapshot()

    def subscribe_stats(self) -> tuple[int, "queue.Queue[StatsSnapshot]"] | None:
        """Register a new live-stats subscriber.

        Returns ``(subscriber_id, queue)`` that will receive one
        :class:`StatsSnapshot` per publisher tick (default 1s). The
        caller MUST call :meth:`unsubscribe_stats` when done — the
        publisher keeps a reference otherwise.

        Returns ``None`` if stats are disabled or no pipeline is
        running.
        """
        if self.stats_counter is None:
            return None
        return self.stats_counter.subscribe()

    def unsubscribe_stats(self, sub_id: int) -> None:
        """Release a subscriber registered via :meth:`subscribe_stats`."""
        if self.stats_counter is None:
            return
        self.stats_counter.unsubscribe(sub_id)

    def _run(self) -> None:
        """Orchestrate the three-stage pipeline.

        Reader → Inferer → Output run as separate threads. They communicate
        exclusively through :class:`queue.Queue` instances; the Inferer→
        Output queue uses ``put_nowait`` so a slow Output thread cannot
        stall the GPU (frames are dropped instead).
        """
        # Queue sizing rationale:
        # - q_infer (Reader→Inferer): small (1-2). Inference is the slow
        # stage; we don't want to buffer raw frames for long because each
        # 1280x720 BGR frame is ~2.7MB.
        # - q_out (Inferer→Output): slightly larger (2-4) so the output
        # stage has room to absorb short render hiccups without dropping.
        # The exact size is exposed as ``infer_to_output_queue_size`` on
        # the AppConfig later if needed; for now hardcode sensible
        # defaults.
        q_infer: queue.Queue = queue.Queue(maxsize=2)
        q_out: queue.Queue = queue.Queue(maxsize=4)

        ctx = PipelineCtx()
        self.ctx = ctx

        try:
            self.source = FFmpegSource(self.cfg.source)
            self.detector = YoloDetector(self.cfg.detector)
            self.tracker = ByteTrackAdapter(self.cfg.tracker)
            self.renderer = Renderer(self.cfg.render)
            self.sink = self.sink or FFmpegSrsSink(self.cfg.sink)

            self.source.start()
            self.sink.start()
            self.status = PipelineStatus.RUNNING
            LOGGER.info(
                "Pipeline started task_id=%s stream=%s (decoupled3-stage)",
                self.task_id,
                self.cfg.sink.stream_name,
            )

            # Three worker threads, joined in dependency order on shutdown.
            # Reader must drain before Inferer can finish; Inferer before
            # Output. We use daemon threads so a hung worker cannot block
            # process exit.
            t_reader = threading.Thread(
                target=reader_worker,
                name=f"mot-reader-{self.task_id}",
                args=(ctx, self.source, q_infer),
                daemon=True,
            )
            t_inferer = threading.Thread(
                target=inferer_worker,
                name=f"mot-inferer-{self.task_id}",
                args=(ctx, self.detector, self.tracker, self.stats_counter, q_infer, q_out, 4),
                daemon=True,
            )
            t_output = threading.Thread(
                target=output_worker,
                name=f"mot-output-{self.task_id}",
                args=(ctx, self.renderer, self.sink, q_out),
                daemon=True,
            )

            t_reader.start()
            t_inferer.start()
            t_output.start()

            # Main loop just waits for stop / error. Workers do the actual
            # work.
            last_stats_log = time.time()
            STATS_PERIOD_SEC = 10.0
            while not self._stop_event.is_set() and not ctx.should_stop:
                if ctx.error is not None:
                    # Any worker reported a fatal error — break out and let
                    # ``finally`` clean up.
                    break

                if time.time() - last_stats_log >= STATS_PERIOD_SEC:
                    snap = ctx.snapshot()
                    if snap["dropped"] > 0 or snap["read_failures"] > 0:
                        LOGGER.info("pipeline counters: %s", snap)
                    last_stats_log = time.time()

                time.sleep(0.2)

            # Cooperative shutdown: signal stop and let workers unwind.
            ctx.request_stop()
            # Give workers up to 5s to finish cleanly. After that, daemon
            # threads will be killed at process exit — acceptable because
            # they own no critical resources outside source/sink which we
            # stop below.
            for t in (t_reader, t_inferer, t_output):
                t.join(timeout=5)

            # If a worker reported an error, surface it.
            if ctx.error is not None:
                self.error = ctx.error
                self.status = PipelineStatus.ERROR
                LOGGER.error("Pipeline ended in error: %s", ctx.error)
        except Exception as exc:
            self.error = str(exc)
            self.status = PipelineStatus.ERROR
            LOGGER.exception("Pipeline failed")
        finally:
            self._release()
            if self.status == PipelineStatus.STOPPING:
                self.status = PipelineStatus.STOPPED
            final = self.ctx.snapshot() if self.ctx is not None else {}
            LOGGER.info(
                "Pipeline finished task_id=%s status=%s counters=%s",
                self.task_id,
                self.status,
                final,
            )

    def _release(self) -> None:
        if self.stats_reporter is not None:
            try:
                self.stats_reporter.stop()
            except Exception:
                LOGGER.exception("Failed to stop stats reporter")
        if self.stats_publisher is not None:
            try:
                self.stats_publisher.stop()
            except Exception:
                LOGGER.exception("Failed to stop stats publisher")
        for component in (self.source, self.sink):
            if component is None:
                continue
            try:
                component.stop()
            except Exception:
                LOGGER.exception("Failed to stop component %s", type(component).__name__)
