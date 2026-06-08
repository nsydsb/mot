from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.bytetrack.iou import bbox_ious


def iou_distance(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    return 1.0 - bbox_ious(track_boxes, det_boxes)


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind, strict=False):
        if cost_matrix[r, c] <= thresh:
            matches.append((int(r), int(c)))
            unmatched_rows.discard(int(r))
            unmatched_cols.discard(int(c))
    return matches, sorted(unmatched_rows), sorted(unmatched_cols)
