"""Smoke test for the DETR postprocess (logits + normalized cxcywh -> [N,6] aabb).

Pure NumPy, no torch. Verifies:

* shape and dtype of the output
* correct cxcywh -> xyxy conversion in pixel space
* softmax "no-object" column is dropped
* confidence / class whitelist filters behave as documented

Run with:

    python -m detectors.detr.tests.test_postprocess
"""

from __future__ import annotations

import numpy as np

from detectors.detr.postprocess import detr_outputs_to_detections


def _logits_class0(p: float, c: int = 3) -> np.ndarray:
    """Build a ``[C+1]`` logit vector where softmax over the **non-background**
    classes gives ``class 0`` probability ≈ ``p``.

    Implemented as ``log(p / (1-p))`` injected against a uniform background
    of the other ``C`` classes (and the trailing no-object class) — this
    keeps the math in line with softmax, not sigmoid.
    """
    out = np.zeros((c + 1,), dtype=np.float32)
    out[0] = float(np.log(max(p, 1e-6) / max(1 - p, 1e-6)))
    return out


def test_empty_output_when_all_below_threshold() -> None:
    # p=0.05 -> logit ~ -2.94 vs uniform background logit 0 across 3 classes
    # -> softmax ~ 0.05, well under the 0.5 gate.
    logits = np.stack([_logits_class0(0.05, c=2) for _ in range(5)], axis=0)
    boxes = np.zeros((5, 4), dtype=np.float32)
    out = detr_outputs_to_detections(
        logits=logits, boxes=boxes, image_size=(100, 100), conf_threshold=0.5
    )
    assert out.shape == (0, 6)
    assert out.dtype == np.float32


def test_cxcywh_to_xyxy_in_pixel_space() -> None:
    # One detection: cx=0.5, cy=0.5, w=0.2, h=0.4 — all normalized.
    # On a (200, 100) image: cx=100, cy=50, w=40, h=40 -> xyxy (80, 30, 120, 70).
    logits = np.stack([_logits_class0(0.99, c=2)], axis=0)
    boxes = np.asarray([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32)
    out = detr_outputs_to_detections(
        logits=logits, boxes=boxes, image_size=(200, 100), conf_threshold=0.5
    )
    assert out.shape == (1, 6)
    x1, y1, x2, y2, score, cls = out[0]
    assert abs(x1 - 80) < 1e-3
    assert abs(y1 - 30) < 1e-3
    assert abs(x2 - 120) < 1e-3
    assert abs(y2 - 70) < 1e-3
    assert score > 0.9
    assert int(cls) == 0


def test_class_whitelist_filters() -> None:
    # 3 detections: class 0 (high), class 1 (high), class 0 (low).
    logits = np.stack(
        [
            _logits_class0(0.99, c=2),  # class 0
            _logits_with_class(cls=1, p=0.95, c=2),  # class 1
            _logits_class0(0.99, c=2),  # class 0 again
        ],
        axis=0,
    )
    boxes = np.asarray(
        [
            [0.2, 0.2, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.8, 0.8, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    out_all = detr_outputs_to_detections(
        logits=logits, boxes=boxes, image_size=(100, 100), conf_threshold=0.5
    )
    assert out_all.shape == (3, 6), out_all

    out_only_cls1 = detr_outputs_to_detections(
        logits=logits,
        boxes=boxes,
        image_size=(100, 100),
        conf_threshold=0.5,
        target_classes=[1],
    )
    assert out_only_cls1.shape == (1, 6), out_only_cls1
    assert int(out_only_cls1[0, 5]) == 1


def test_query_count_mismatch_raises() -> None:
    logits = np.zeros((3, 4), dtype=np.float32)
    boxes = np.zeros((5, 4), dtype=np.float32)
    try:
        detr_outputs_to_detections(logits=logits, boxes=boxes, image_size=(10, 10))
    except ValueError:
        return
    raise AssertionError("expected ValueError on query count mismatch")


def _logits_with_class(cls: int, p: float, c: int) -> np.ndarray:
    """Same construction as :func:`_logits_class0` but targeting a
    specific class index instead of 0."""
    out = np.zeros((c + 1,), dtype=np.float32)
    out[cls] = float(np.log(max(p, 1e-6) / max(1 - p, 1e-6)))
    return out


def main() -> None:
    tests = [
        test_empty_output_when_all_below_threshold,
        test_cxcywh_to_xyxy_in_pixel_space,
        test_class_whitelist_filters,
        test_query_count_mismatch_raises,
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
    print("All postprocess tests passed.")


if __name__ == "__main__":
    main()
