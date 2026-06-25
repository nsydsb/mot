from __future__ import annotations

import numpy as np


def detr_outputs_to_detections(
    logits: np.ndarray,
    boxes: np.ndarray,
    image_size: tuple[int, int],
    conf_threshold: float = 0.5,
    target_classes: list[int] | None = None,
) -> np.ndarray:
    """Convert raw DETR-style model outputs into the project's ``[N, 6]`` aabb format.

    Parameters
    ----------
    logits:
        ``[Q, C+1]`` class logits, where the last column is the
        "no-object" class. Softmax is applied along the last axis.
    boxes:
        ``[Q, 4]`` boxes in **normalized cxcywh** (DETR / Deformable
        DETR convention). HF ``transformers`` returns cxcywh already
        scaled to ``[0, 1]``.
    image_size:
        ``(width, height)`` of the original frame the boxes refer to.
    conf_threshold:
        Minimum softmax probability (over non-"no-object" classes)
        to keep.
    target_classes:
        Optional whitelist of class ids to keep. ``None`` keeps all
        non-background classes.

    Returns
    -------
    ``[N, 6]`` float32 array ``[x1, y1, x2, y2, score, cls]`` in the
    image's pixel coordinate system. Empty input (or all below
    threshold) yields ``[0, 6]``.
    """
    logits = np.asarray(logits, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if logits.ndim != 2 or boxes.ndim != 2:
        raise ValueError(
            f"Expected 2-D logits/boxes, got logits {logits.shape} / boxes {boxes.shape}"
        )
    if logits.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"Query count mismatch: logits {logits.shape[0]} vs boxes {boxes.shape[0]}"
        )
    if boxes.shape[1] != 4:
        raise ValueError(f"Boxes must be [Q, 4] cxcywh, got {boxes.shape}")

    probs = _softmax(logits, axis=-1)
    obj_probs = probs[:, :-1]
    scores = obj_probs.max(axis=-1)
    classes = obj_probs.argmax(axis=-1)

    keep = scores >= float(conf_threshold)
    if target_classes is not None:
        target_set = {int(c) for c in target_classes}
        keep &= np.isin(classes, list(target_set))
    if not np.any(keep):
        return np.empty((0, 6), dtype=np.float32)

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    cx = boxes[:, 0] * image_size[0]
    cy = boxes[:, 1] * image_size[1]
    w = boxes[:, 2] * image_size[0]
    h = boxes[:, 3] * image_size[1]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    out = np.stack([x1, y1, x2, y2, scores, classes.astype(np.float32)], axis=1)
    return out.astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
