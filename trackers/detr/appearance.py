"""Appearance-only tracking primitives used by the DETR tracker.

Anything that involves an embedding lives here, because appearance is
what separates ``trackers/detr`` from the motion-only trackers
(``trackers/bytetrack``). Exposed:

* :func:`l2_normalize`, :func:`cosine_distance` — cheap, pure NumPy.
* :class:`FeatureBank` — EMA + bounded ring of recent embeddings per
  track (BoxMOT-style).
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1-D or 2-D array along the last axis."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr) + eps
        return (arr / norm).astype(np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True) + eps
    return (arr / norm).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine distance ``1 - cos_sim``, ``[len(a), len(b)]`` float32.

    Inputs are L2-normalized internally. Empty inputs produce a
    correctly-shaped zero matrix.
    """
    a_n = l2_normalize(a, eps=eps)
    b_n = l2_normalize(b, eps=eps)
    if a_n.size == 0 or b_n.size == 0:
        return np.zeros((a_n.shape[0], b_n.shape[0]), dtype=np.float32)
    sim = a_n @ b_n.T
    return (1.0 - sim).astype(np.float32)


def pad_or_trim(feat: np.ndarray, dim: int) -> np.ndarray:
    """Pad with zeros or trim to ``dim`` along the last axis.

    Used to align embeddings whose source models disagree on the
    output width (e.g. swin vs resnet backbones).
    """
    arr = np.asarray(feat, dtype=np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] > dim:
        return arr[:dim]
    out = np.zeros((dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


class FeatureBank:
    """Bounded appearance embedding bank for a single track.

    Keeps a small ring of recent embeddings (default 100) and an EMA
    mean. Matching code uses the EMA mean as the track's identity.
    """

    def __init__(self, max_size: int = 100, ema_alpha: float = 0.9) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.max_size = int(max_size)
        self.ema_alpha = float(ema_alpha)
        self._items: list[np.ndarray] = []
        self._mean: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def mean(self) -> np.ndarray | None:
        if self._mean is None or not self._items:
            return None
        return self._mean.copy()

    def reset(self) -> None:
        self._items.clear()
        self._mean = None

    def update(self, feat: np.ndarray) -> None:
        arr = np.asarray(feat, dtype=np.float32).reshape(-1)
        if self._mean is None:
            self._mean = arr.copy()
        else:
            self._mean = self.ema_alpha * self._mean + (1.0 - self.ema_alpha) * arr
        self._items.append(arr)
        if len(self._items) > self.max_size:
            self._items.pop(0)

    def all(self) -> np.ndarray:
        if not self._items:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(self._items, axis=0)
