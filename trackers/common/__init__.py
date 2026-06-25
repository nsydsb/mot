"""Common tracking primitives shared by every concrete tracker.

This package is the public vocabulary of the project's tracker layer.
Anything that more than one tracker needs lives here; anything
algorithm-specific stays inside the tracker's own folder
(``trackers/bytetrack/``, ``trackers/detr/``, ...).

Currently exposed:

* :class:`BaseTrack`, :class:`TrackState` — monotonic id allocator and
  state machine used by every track.
* :class:`Detection` — a single observation handed to the tracker
  (box + score + class + slot + optional embedding).
* :func:`xyxy_to_xyah`, :func:`xyah_to_xyxy` — coordinate conversions
  used by the constant-velocity Kalman filter.
* :class:`KalmanFilter` — xyah constant-velocity KF.
* :func:`bbox_ious`, :func:`iou_distance` — pairwise box geometry.
* :func:`linear_assignment` — gated Hungarian matching.

New trackers should depend on this package, not on each other.
"""

from __future__ import annotations

from trackers.common.basetrack import BaseTrack, TrackState
from trackers.common.detection import (
    Detection,
    xyxy_to_xyah,
    xyah_to_xyxy,
)
from trackers.common.iou import bbox_ious, iou_distance
from trackers.common.kalman_filter import KalmanFilter
from trackers.common.matching import linear_assignment

__all__ = [
    "BaseTrack",
    "Detection",
    "KalmanFilter",
    "TrackState",
    "bbox_ious",
    "iou_distance",
    "linear_assignment",
    "xyxy_to_xyah",
    "xyah_to_xyxy",
]
