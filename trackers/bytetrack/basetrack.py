from __future__ import annotations

from enum import Enum


class TrackState(str, Enum):
    Tracked = "Tracked"
    Lost = "Lost"
    Removed = "Removed"


class BaseTrack:
    _count = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count
