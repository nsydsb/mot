from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.manager import DetectionManager
from logging_utils.setup import get_ring_handler

LOGGER = logging.getLogger(__name__)


class StartRequest(BaseModel):
    source_url: str | None = None
    stream_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    model_type: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    # Restrict detection to specific classes. Accepts either COCO class ids
    # (e.g. [2, 5, 7]) or class names (e.g. ["car", "bus", "truck"]).
    # Names are resolved against the loaded YOLO model.
    classes: list[int | str] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


def create_router(manager: DetectionManager) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/detection/start")
    def start_detection(req: StartRequest) -> dict[str, Any]:
        try:
            state = manager.start(
                req.source_url,
                req.stream_name,
                req.model_type,
                req.classes,
                req.overrides,
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "task_id": state.task_id,
            "status": state.status,
            "stream_name": state.stream_name,
            "play_urls": state.play_urls,
        }

    @router.post("/v1/detection/stop")
    def stop_detection() -> dict[str, Any]:
        state = manager.stop()
        return {"task_id": state.task_id, "status": state.status}

    @router.get("/v1/detection/stats")
    def detection_stats() -> dict[str, Any]:
        view = manager.stats()
        if view is None:
            raise HTTPException(
                status_code=409,
                detail="No active detection task or stats are disabled",
            )
        return view

    @router.get("/v1/logs")
    def logs(tail: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        return {"logs": get_ring_handler().tail(tail)}

    @router.websocket("/v1/detection/stats/ws")
    async def detection_stats_ws(websocket: WebSocket) -> None:
        """Stream in-scene track counts to the frontend at ~1Hz.

        Wire protocol (server -> client, one JSON message per tick):
            {
              "type": "stats",
              "task_id": "...",
              "stream_name": "...",
              "total": int,
              "window_sec": float,
              "categories": {"person": 3, "car": 5, ...},
              "tracked_ids": {"person": [1, 7, 9], ...},
              "taken_at": float
            }

        Heartbeats (``{"type": "heartbeat"}``) are sent if no snapshot
        arrives within 1s, so the client can distinguish a quiet scene
        from a dead connection. The server closes the socket when the
        pipeline is stopped (the publisher pushes a ``None`` sentinel
        into every subscriber queue).
        """
        await websocket.accept()
        pipeline = manager.current_pipeline()
        if pipeline is None:
            await websocket.send_json(
                {"type": "error", "message": "No active detection task"}
            )
            await websocket.close(code=1011)
            return
        sub = pipeline.subscribe_stats()
        if sub is None:
            await websocket.send_json(
                {"type": "error", "message": "Stats are disabled in config"}
            )
            await websocket.close(code=1011)
            return
        sub_id, snap_queue = sub
        LOGGER.info("stats ws connected sub_id=%d", sub_id)
        loop = asyncio.get_running_loop()
        try:
            while True:
                # Bridge the thread-safe Queue from the publisher thread
                # into an asyncio consumer without blocking the event
                # loop. A short timeout lets us emit heartbeats and
                # notice dead pipelines.
                item = await loop.run_in_executor(
                    None, _drain_subscriber_queue, snap_queue, 1.0
                )
                if item is None:
                    # Publisher pushed a None sentinel — pipeline is
                    # stopping. Close cleanly.
                    break
                if item is _NO_DATA_SENTINEL:
                    # Quiet tick — send a heartbeat so the client knows
                    # we're still here. Useful when the inferer has no
                    # detections (e.g. empty frames) and thus no
                    # counter updates have happened.
                    await websocket.send_json({"type": "heartbeat"})
                    continue
                snap = item
                await websocket.send_json(
                    {
                        "type": "stats",
                        "task_id": pipeline.task_id,
                        "stream_name": pipeline.cfg.sink.stream_name,
                        "total": snap.total,
                        "window_sec": snap.window_sec,
                        "categories": snap.counts,
                        "tracked_ids": snap.tracked_ids,
                        "taken_at": snap.taken_at,
                    }
                )
        except WebSocketDisconnect:
            LOGGER.info("stats ws disconnected sub_id=%d", sub_id)
        except Exception:
            LOGGER.exception("stats ws error sub_id=%d", sub_id)
        finally:
            pipeline.unsubscribe_stats(sub_id)
            try:
                await websocket.close()
            except Exception:
                pass

    return router


# Sentinels used by the WebSocket bridge. Defined module-level so the
# helper function below can return them by identity.
# ``None`` is reserved as the "pipeline shutting down" signal pushed
# by :meth:`TrackCategoryCounter.close_all_subscribers`; real
# snapshots are never None.
_NO_DATA_SENTINEL = object()


def _drain_subscriber_queue(q, timeout: float):
    """Pop one item from a subscriber queue, or signal a quiet tick.

    Returns the popped item (``None`` means "publisher closed the
    subscriber, exit the loop"), ``_NO_DATA_SENTINEL`` on timeout, or
    a real ``StatsSnapshot`` instance.
    """
    try:
        return q.get(timeout=timeout)
    except Exception:
        return _NO_DATA_SENTINEL
