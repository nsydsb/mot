from __future__ import annotations

from enum import Enum


class TrackState(str, Enum):
    New = "New"
    Tracked = "Tracked"
    Lost = "Lost"
    Removed = "Removed"


class BaseTrack:
    """Base class for every concrete track in the project.

    Provides a monotonic global id allocator and the lifecycle helpers
    that every tracker in the project agrees on. Concrete trackers
    (BYTETrack's ``STrack``, DetrTracker's ``STrack``, ...) inherit
    from this and add their own state fields (Kalman mean/covariance,
    appearance bank, hits counter, etc.).

    Keeping the id allocator here means a track created by one tracker
    and a track created by another cannot collide — handy when we
    eventually wire multiple trackers into the same scene.
    """

    _count = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed
