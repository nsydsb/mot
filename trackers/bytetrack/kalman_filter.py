"""Backwards-compat shim.

The canonical implementation now lives in :mod:`trackers.common.kalman_filter`.
"""

from __future__ import annotations

from trackers.common.kalman_filter import KalmanFilter

__all__ = ["KalmanFilter"]
