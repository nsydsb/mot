from __future__ import annotations

import numpy as np

from config.schema import TrackerConfig
from trackers.bytetrack.bytetrack import BYTETracker


class ByteTrackAdapter:
    def __init__(self, cfg: TrackerConfig):
        self.tracker = BYTETracker(cfg)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        _ = frame
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] != 6:
            raise ValueError(f"Expected detector output shape [N, 6], got {dets.shape}")
        tracks = self.tracker.update(dets.astype(np.float32, copy=False))
        if not tracks:
            return np.empty((0, 8), dtype=np.float32)
        rows = [
            [*t.tlbr.tolist(), float(t.track_id), t.score, t.cls, float(t.det_ind)]
            for t in tracks
        ]
        return np.asarray(rows, dtype=np.float32)
