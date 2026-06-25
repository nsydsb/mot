"""Offline demo for the DETR tracker.

Doesn't need torch / transformers / a real model — it synthesizes a
random frame, fabricates ``[N, 6]`` detections + random embeddings,
and runs :class:`trackers.detr.detr_tracker.DetrTracker` end-to-end.

Run with:

    python -m trackers.detr.demo
"""

from __future__ import annotations

import numpy as np

from trackers.detr.detr_tracker import DetrTracker, DetrTrackerConfig


def _synth_frame(h: int = 320, w: int = 480, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _fake_detections(frame: np.ndarray, n: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    boxes: list[list[float]] = []
    for _ in range(n):
        bw = float(rng.uniform(40, 90))
        bh = float(rng.uniform(60, 120))
        cx = float(rng.uniform(bw, w - bw))
        cy = float(rng.uniform(bh, h - bh))
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 0.9, 0.0])
    return np.asarray(boxes, dtype=np.float32)


def _fake_embeddings(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feats = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats


def main() -> None:
    cfg = DetrTrackerConfig(
        track_thresh=0.4,
        low_thresh=0.1,
        match_thresh=0.8,
        second_match_thresh=0.6,
        track_buffer=15,
        min_hits=1,  # permissive so we see ids on frame 1
    )
    tracker = DetrTracker(cfg)

    for frame_idx in range(4):
        frame = _synth_frame(seed=frame_idx)
        dets = _fake_detections(frame, n=2, seed=frame_idx)
        embs = _fake_embeddings(n=dets.shape[0], dim=16, seed=frame_idx)
        tracks = tracker.update(dets, frame=frame, embeddings=embs)
        print(
            f"frame={frame_idx:02d} dets={dets.shape[0]} tracks={tracks.shape[0]} "
            f"ids={tracks[:, 4].astype(int).tolist() if tracks.size else []}"
        )


if __name__ == "__main__":
    main()
