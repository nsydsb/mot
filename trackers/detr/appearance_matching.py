"""Cost fusion for trackers that combine motion and appearance.

Only the appearance-bearing trackers (currently ``trackers/detr``)
need this; motion-only trackers keep using :func:`iou_distance`
directly from :mod:`trackers.common.iou`.
"""

from __future__ import annotations

import numpy as np


def fuse_motion_appearance(
    iou_cost: np.ndarray, app_cost: np.ndarray, lambda_iou: float = 0.5
) -> np.ndarray:
    """BoxMOT-style fused cost.

    ``cost = lambda_iou * iou_cost + (1 - lambda_iou) * app_cost``

    Both inputs are expected to use ``smaller = closer`` semantics
    and be in ``[0, 1+]``. The gating happens downstream in
    :func:`trackers.common.matching.linear_assignment`.
    """
    if iou_cost.size == 0:
        return iou_cost
    if iou_cost.shape != app_cost.shape:
        raise ValueError(
            f"Cost shape mismatch: iou {iou_cost.shape} vs appearance {app_cost.shape}"
        )
    lambda_iou = float(np.clip(lambda_iou, 0.0, 1.0))
    return (lambda_iou * iou_cost + (1.0 - lambda_iou) * app_cost).astype(np.float32)
