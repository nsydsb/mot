from __future__ import annotations

import numpy as np


def bbox_ious(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy box arrays.

    Empty inputs produce a correctly-shaped zero matrix instead of
    raising. This is the canonical box-geometry primitive for the
    tracker layer; both ``trackers/bytetrack`` and ``trackers/detr``
    route through this implementation now.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def iou_distance(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """``1 - IoU`` distance matrix in the convention used by every
    tracker in this project (smaller is closer / cheaper)."""
    return 1.0 - bbox_ious(track_boxes, det_boxes)
