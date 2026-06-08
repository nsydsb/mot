from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PipelineStatus(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class PipelineState:
    task_id: str | None = None
    status: PipelineStatus = PipelineStatus.IDLE
    stream_name: str | None = None
    play_urls: dict[str, str] = field(default_factory=dict)
    error: str | None = None
