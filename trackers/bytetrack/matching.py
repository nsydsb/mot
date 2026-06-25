"""Backwards-compat shim.

The canonical implementation now lives in :mod:`trackers.common.matching`.
The DETR tracker adds a ``fuse_motion_appearance`` helper that lives in
``trackers/detr/appearance_matching.py`` because it's only relevant for
appearance-bearing trackers.
"""

from __future__ import annotations

from trackers.common.matching import linear_assignment

__all__ = ["linear_assignment"]
