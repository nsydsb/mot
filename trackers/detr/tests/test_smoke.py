"""Contract smoke test for the DETR tracker.

Verifies, **without** torch / transformers:

* :class:`trackers.detr.detr_tracker.DetrTracker` is importable.
* ``update`` returns ``[N, 8]`` aabb (the project's standard tracker
  output format).
* ids stay consistent across consecutive frames when detections are
  stable.
* The second-stage refind keeps the id when a track temporarily drops
  to a low score (BoxMOT-style).

Run with:

    python -m trackers.detr.tests.test_smoke
"""

from __future__ import annotations

import numpy as np

from trackers.detr.detr_tracker import DetrTracker, DetrTrackerConfig


def _box(cx: float, cy: float, w: float = 60.0, h: float = 80.0) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, 0.0]


def _det(box: list[float], emb_dim: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    det = np.asarray(box, dtype=np.float32).reshape(1, 6)
    emb = rng.normal(0, 1, size=(1, emb_dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return det, emb


def test_import_without_torch() -> None:
    tracker = DetrTracker(DetrTrackerConfig())
    assert tracker.frame_id == 0


def test_empty_input_returns_empty_output() -> None:
    trk = DetrTracker(DetrTrackerConfig(min_hits=1))
    out = trk.update(np.empty((0, 6), dtype=np.float32))
    assert out.shape == (0, 8)
    assert out.dtype == np.float32


def test_id_consistency_across_frames() -> None:
    cfg = DetrTrackerConfig(track_thresh=0.4, low_thresh=0.1, min_hits=1)
    trk = DetrTracker(cfg)
    det1, emb1 = _det(_box(cx=200, cy=150), seed=1)
    out1 = trk.update(det1, embeddings=emb1)
    assert out1.shape == (1, 8), out1
    id1 = int(out1[0, 4])

    det2, emb2 = _det(_box(cx=210, cy=155), seed=2)  # small move
    out2 = trk.update(det2, embeddings=emb2)
    assert out2.shape == (1, 8)
    assert int(out2[0, 4]) == id1, "track id should remain stable"


def test_lost_then_refind_via_low_score() -> None:
    cfg = DetrTrackerConfig(
        track_thresh=0.5,
        low_thresh=0.1,
        match_thresh=0.8,
        second_match_thresh=0.6,
        track_buffer=5,
        min_hits=1,
    )
    trk = DetrTracker(cfg)
    det_high, emb = _det(_box(cx=200, cy=150), seed=1)
    out1 = trk.update(det_high, embeddings=emb)
    assert out1.shape[0] == 1
    track_id = int(out1[0, 4])

    # Frame 2: same object, but the detection drops to a low score
    # (still in [low_thresh, track_thresh)). The track should be
    # refound via the second stage and keep its id.
    det_low = np.asarray(
        [[200 - 30, 150 - 40, 200 + 30, 150 + 40, 0.3, 0.0]],
        dtype=np.float32,
    )
    out2 = trk.update(det_low, embeddings=emb)
    assert out2.shape[0] == 1, out2
    assert int(out2[0, 4]) == track_id, "low-score refind should keep id"


def test_output_is_aabb_xyxy() -> None:
    trk = DetrTracker(DetrTrackerConfig(min_hits=1))
    det, emb = _det(_box(cx=300, cy=200), seed=3)
    out = trk.update(det, embeddings=emb)
    assert out.shape == (1, 8)
    x1, y1, x2, y2, tid, score, cls, det_ind = out[0]
    assert x2 > x1 and y2 > y1, "must be aabb xyxy"


def test_appearance_cost_shape() -> None:
    """Smoke test on the appearance sub-pipeline specifically."""
    cfg = DetrTrackerConfig(min_hits=1, allow_embeddingless=False)
    trk = DetrTracker(cfg)
    det = np.asarray([_box(cx=100, cy=100)], dtype=np.float32)
    emb = np.eye(4, dtype=np.float32)[:1]  # shape (1, 4)
    # If allow_embeddingless=False and we don't pass embeddings it must raise.
    try:
        trk.update(det)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when embeddings missing")
    # With embeddings it should work and produce 8 columns.
    out = trk.update(det, embeddings=emb)
    assert out.shape == (1, 8)


def main() -> None:
    tests = [
        test_import_without_torch,
        test_empty_input_returns_empty_output,
        test_id_consistency_across_frames,
        test_lost_then_refind_via_low_score,
        test_output_is_aabb_xyxy,
        test_appearance_cost_shape,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"PASS  {t.__name__}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
