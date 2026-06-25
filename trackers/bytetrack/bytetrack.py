from __future__ import annotations

import numpy as np

from config.schema import TrackerConfig
from trackers.common import (
    BaseTrack,
    Detection,
    KalmanFilter,
    TrackState,
    iou_distance,
    linear_assignment,
    xyah_to_xyxy,
    xyxy_to_xyah,
)


class STrack(BaseTrack):
    def __init__(self, det: Detection):
        self.tlbr = det.tlbr.astype(np.float32)
        self.score = float(det.score)
        self.cls = float(det.cls)
        self.det_ind = int(det.det_ind)
        self.track_id = 0
        self.state = TrackState.Tracked
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(xyxy_to_xyah(self.tlbr))
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.state = TrackState.Tracked

    def predict(self, kalman_filter: KalmanFilter) -> None:
        if self.mean is None or self.covariance is None:
            return
        self.mean, self.covariance = kalman_filter.predict(self.mean, self.covariance)
        self.tlbr = xyah_to_xyxy(self.mean[:4])
        self.time_since_update += 1

    def update(self, det: Detection, kalman_filter: KalmanFilter, frame_id: int) -> None:
        if self.mean is None or self.covariance is None:
            self.activate(kalman_filter, frame_id)
        else:
            self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, xyxy_to_xyah(det.tlbr))
        self.tlbr = det.tlbr.astype(np.float32)
        self.score = float(det.score)
        self.cls = float(det.cls)
        self.det_ind = int(det.det_ind)
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = TrackState.Tracked

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


class BYTETracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0

    def update(self, dets: np.ndarray) -> list[STrack]:
        self.frame_id += 1
        detections = [
            Detection(
                tlbr=d[:4],
                score=float(d[4]),
                cls=float(d[5]),
                det_ind=i,
                # BYTETrack is motion-only; appearance is intentionally None.
                embedding=None,
            )
            for i, d in enumerate(dets)
            if (d[2] - d[0]) * (d[3] - d[1]) >= self.cfg.min_box_area
        ]
        high = [d for d in detections if d.score >= self.cfg.track_thresh]
        low = [d for d in detections if self.cfg.low_thresh <= d.score < self.cfg.track_thresh]

        strack_pool = [t for t in self.tracked_stracks + self.lost_stracks if t.state != TrackState.Removed]
        for track in strack_pool:
            track.predict(self.kalman_filter)

        activated: list[STrack] = []
        lost: list[STrack] = []

        matches, u_track, u_det = self._match(strack_pool, high, self.cfg.match_thresh)
        for ti, di in matches:
            strack_pool[ti].update(high[di], self.kalman_filter, self.frame_id)
            activated.append(strack_pool[ti])

        remaining_tracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        matches2, u_track2, _ = self._match(remaining_tracks, low, 0.5)
        for ti, di in matches2:
            remaining_tracks[ti].update(low[di], self.kalman_filter, self.frame_id)
            activated.append(remaining_tracks[ti])

        for idx in u_track2:
            remaining_tracks[idx].mark_lost()
            lost.append(remaining_tracks[idx])

        for idx in u_det:
            track = STrack(high[idx])
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        live_ids = {t.track_id for t in activated}
        self.tracked_stracks = [t for t in self.tracked_stracks if t.track_id in live_ids]
        for t in activated:
            if t.track_id not in {x.track_id for x in self.tracked_stracks}:
                self.tracked_stracks.append(t)

        self.lost_stracks = [t for t in self.lost_stracks + lost if t.track_id not in live_ids]
        kept_lost: list[STrack] = []
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.cfg.track_buffer:
                t.mark_removed()
                self.removed_stracks.append(t)
            elif t.state != TrackState.Removed:
                kept_lost.append(t)
        self.lost_stracks = kept_lost
        return [t for t in self.tracked_stracks if t.state == TrackState.Tracked and t.time_since_update == 0]

    @staticmethod
    def _match(tracks: list[STrack], detections: list[Detection], match_thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float32)
        det_boxes = np.asarray([d.tlbr for d in detections], dtype=np.float32)
        cost = iou_distance(track_boxes, det_boxes)
        return linear_assignment(cost, match_thresh)
