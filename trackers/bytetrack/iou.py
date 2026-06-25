"""Backwards-compat shim.

The canonical implementation now lives in :mod:`trackers.common.iou`.
"""

from __future__ import annotations

from trackers.common.iou import bbox_ious, iou_distance

__all__ = ["bbox_ious", "iou_distance"]
