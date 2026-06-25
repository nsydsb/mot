from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def xyxy_to_xyah(xyxy: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` to ``[cx, cy, a, h]`` (xyah).

    The aspect ratio ``a = w / h`` is clamped to ``h >= 1e-6`` so a
    degenerate zero-height box still produces a finite ratio.
    """
    w = float(xyxy[2]) - float(xyxy[0])
    h = float(xyxy[3]) - float(xyxy[1])
    x = float(xyxy[0]) + w / 2.0
    y = float(xyxy[1]) + h / 2.0
    return np.array([x, y, w / max(h, 1e-6), h], dtype=np.float32)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Inverse of :func:`xyxy_to_xyah`. Takes ``[cx, cy, a, h]``."""
    x, y, a, h = float(xyah[0]), float(xyah[1]), float(xyah[2]), float(xyah[3])
    w = a * h
    return np.array([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], dtype=np.float32)


@dataclass
class Detection:
    """A single observation handed to a tracker.

    Holds the geometric part of a detection (``tlbr``), the metadata
    the project requires on every track (``score``, ``cls``,
    ``det_ind``), and an optional appearance embedding. The embedding
    is allowed to be ``None``; trackers that need it will fall back
    to motion-only matching in that case (see
    ``trackers.detr.detr_tracker.DetrTracker``).
    """

    tlbr: np.ndarray
    score: float
    cls: float
    det_ind: int
    embedding: np.ndarray | None = None
