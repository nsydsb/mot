"""Backwards-compat shim.

The canonical implementation now lives in :mod:`trackers.common`. This
module is kept so existing imports (``from trackers.bytetrack.basetrack
import BaseTrack, TrackState``) keep working unchanged. New code should
import from :mod:`trackers.common` directly.
"""

from __future__ import annotations

from trackers.common.basetrack import BaseTrack, TrackState

__all__ = ["BaseTrack", "TrackState"]
