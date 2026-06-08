from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.manager import DetectionManager
from logging_utils.setup import get_ring_handler


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

    return router
