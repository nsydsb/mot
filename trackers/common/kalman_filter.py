from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Constant-velocity Kalman filter in ``[cx, cy, a, h]`` (xyah) form.

    State vector is 8-D (``[cx, cy, a, h, vx, vy, va, vh]``); the
    measurement is 4-D position only. Process / measurement noise
    scale with the current height ``h`` — this matches the convention
    used by the original ByteTrack implementation and is independent
    of the chosen detector (DETR, YOLO, ...).
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self.motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = dt
        self.update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)].astype(np.float32)
        h = float(measurement[3])
        std = [
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * h,
            1e-2,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * h,
            1e-5,
            10 * self.std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std_pos = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ]
        std_vel = [
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T
        innovation_cov = np.diag(np.square([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ])).astype(np.float32)
        projected_cov += innovation_cov
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_cov.astype(np.float32)
