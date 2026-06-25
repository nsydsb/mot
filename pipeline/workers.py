"""Worker loops for the decoupled3-stage pipeline.

Stage graph::

 [Reader] --(raw FramePacket)--> [Inferer] --(RenderJob)--> [Output]
 │ │ │
 source.read detect+track render+sink.write

 Each stage runs on its own thread and communicates exclusively through
 :class:`queue.Queue` instances. The Inferer→Output queue uses
 ``put_nowait`` so the GPU inference thread is never blocked when the
 output (encode + RTMP push) cannot keep up — we drop frames instead,
 which is the right call for a live RTMP broadcast where freshness
 matters more than completeness.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Optional

import numpy as np

from detectors.yolo_detector import YoloDetector
from pipeline.jobs import PipelineCtx, RenderJob
from render.renderer import Renderer
from sink.ffmpeg_srs_sink import FFmpegSrsSink
from source.ffmpeg_source import FFmpegSource
from trackers.adapter import ByteTrackAdapter

LOGGER = logging.getLogger(__name__)


# Sentinel pushed onto each queue to signal "no more jobs, drain and
# exit". Using a unique object (not None) keeps None valid as a real
# payload if we ever need it.
_SHUTDOWN = object()


def _check_stop(ctx: PipelineCtx) -> bool:
    return ctx.should_stop


def reader_worker(
    ctx: PipelineCtx,
    source: FFmpegSource,
    q_infer: queue.Queue,
) -> None:
    """Stage1: pull decoded frames from FFmpeg and hand them to the Inferer.

    Runs the blocking ``source.read()`` in a tight loop. The Inferer's
    bounded queue provides natural backpressure — if Inferer can't keep
    up, this thread blocks on ``q_infer.put`` and the source buffer
    fills, which is harmless. We do NOT drop frames here: if the source
    is faster than Inferer, that's a problem on the Inferer side, not
    the source side.
    """
    LOGGER.info("Reader worker started")
    try:
        while not _check_stop(ctx):
            try:
                pkt = source.read()
            except Exception as exc:
                # ``FFmpegSource.read`` raises on short reads or process death.
                # The source already tries to restart internally; if it still
                # raises, give it a moment and retry.
                ctx.incr("read_failures")
                LOGGER.warning("Source read failed (%s); retrying after delay", exc)
                if _check_stop(ctx):
                    break
                time.sleep(getattr(source.cfg, "reconnect_delay_sec", 2.0))
                continue

            ctx.incr("read")

            # Hand off the FramePacket directly. FramePacket is a frozen
            # dataclass, but ``image`` (a numpy array) is not transitively
            # immutable — workers must NEVER mutate ``pkt.image`` in place;
            # always derive a new array (``image.copy()``) if they need to
            # modify it. The Inferer downstream follows this rule.
            q_infer.put(pkt)
    except BaseException as exc:  # noqa: BLE001 — top-level safety net
        ctx.report_error("reader", exc)
    finally:
        q_infer.put(_SHUTDOWN)
        LOGGER.info("Reader worker finished")


def inferer_worker(
    ctx: PipelineCtx,
    detector: YoloDetector,
    tracker: ByteTrackAdapter,
    stats_counter,
    q_infer: queue.Queue,
    q_out: queue.Queue,
    output_queue_size: int,
) -> None:
    """Stage2: detect + track, then enqueue a RenderJob for the Output.

    The downstream queue uses ``put_nowait`` so a slow Output thread can
    never stall inference — instead we increment ``dropped`` and move on.
    This is the correct behaviour for a live RTMP stream: shipping a
    fresh frame at all times is more valuable than every frame being
    pushed in source order.
    """
    LOGGER.info("Inferer worker started")
    try:
        while not _check_stop(ctx):
            try:
                pkt = q_infer.get(timeout=0.5)
            except queue.Empty:
                continue
            if pkt is _SHUTDOWN:
                break

            try:
                dets = detector.predict(pkt.image)
                tracks = tracker.update(dets, pkt.image)
            except Exception as exc:
                # A single frame's inference failed — log it and skip the frame.
                # We don't want one bad frame to tear down the whole pipeline.
                LOGGER.exception(
                    "Inference failed on frame_id=%d (%s); skipping frame",
                    pkt.frame_id,
                    exc,
                )
                continue

            if stats_counter is not None:
                stats_counter.update(tracks)

            ctx.incr("inferred")

            job = RenderJob(
                frame_id=pkt.frame_id,
                image=pkt.image,
                tracks=tracks,
                enqueued_at=time.time(),
            )
            try:
                q_out.put_nowait(job)
            except queue.Full:
                # Output thread is behind; drop this frame rather than block
                # the GPU. We log at DEBUG to avoid spamming — promote to
                # periodic INFO summary if you want to see drop rate.
                ctx.incr("dropped")
                LOGGER.debug(
                    "Output queue full; dropped frame_id=%d (queue=%d)",
                    job.frame_id,
                    q_out.qsize(),
                )
    except BaseException as exc:  # noqa: BLE001
        ctx.report_error("inferer", exc)
    finally:
        # Always signal downstream so Output can drain and exit even on
        # error.
        q_out.put(_SHUTDOWN)
        LOGGER.info("Inferer worker finished")


def output_worker(
    ctx: PipelineCtx,
    renderer: Renderer,
    sink: FFmpegSrsSink,
    q_out: queue.Queue,
) -> None:
    """Stage3: render + write to FFmpeg sink.

    Drains the Inferer→Output queue, renders each job's image with its
    tracks, and writes the result to the FFmpeg pipe. Blocks on
    ``q_out.get`` so a slow upstream just causes us to wait, never to
    back-pressure Inferer (Inferer drops rather than blocks).
    """
    LOGGER.info("Output worker started")
    try:
        while not _check_stop(ctx):
            try:
                job = q_out.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is _SHUTDOWN:
                # Drain remaining items if any (shouldn't be any after a
                # clean shutdown, but be defensive).
                break

            rendered = renderer.render(job.image, job.tracks)
            try:
                sink.write(rendered)
            except Exception as exc:
                ctx.report_error("output", exc)
                break

            ctx.incr("rendered")
    except BaseException as exc:  # noqa: BLE001
        ctx.report_error("output", exc)
    finally:
        LOGGER.info("Output worker finished")
