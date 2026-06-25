from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
from trackers.detr.appearance import FeatureBank, cosine_distance, pad_or_trim
from trackers.detr.appearance_matching import fuse_motion_appearance


@dataclass
class DetrTrackerConfig:
    """BoxMOT-style tracker hyperparameters.

    Tunables follow StrongSORT / BoT-SORT defaults. The appearance
    weight is intentionally strong (``lambda_iou == 0.5`` -> equal
    motion vs appearance) because DETR-style backends benefit most
    from the appearance term when targets cross.
    """

    min_box_area: float = 10.0
    track_thresh: float = 0.5
    low_thresh: float = 0.1
    # Gate for the **first-stage** matching (Tracked + Lost ↔ high dets).
    match_thresh: float = 0.8
    # Looser gate for the second-stage matching against Lost tracks.
    second_match_thresh: float = 0.6
    track_buffer: int = 30
    # ``cost = lambda_iou * (1 - IoU) + (1 - lambda_iou) * (1 - cos)``.
    lambda_iou: float = 0.5
    # If True, detections without embeddings skip the appearance term
    # and the tracker falls back to IoU-only matching for that row.
    allow_embeddingless: bool = True
    # Per-track appearance bank.
    appearance_bank_size: int = 100
    appearance_ema_alpha: float = 0.9
    # Minimum hits before a New track is promoted to Tracked output.
    min_hits: int = 3


class STrack(BaseTrack):
    """Single tracked object with Kalman state + appearance bank."""

    def __init__(self, det: Detection) -> None:
        self.tlbr = det.tlbr.astype(np.float32)
        self.score = float(det.score)
        self.cls = float(det.cls)
        self.det_ind = int(det.det_ind)
        self.track_id: int = 0
        self.state = TrackState.New
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.frame_id: int = 0
        self.start_frame: int = 0
        self.time_since_update: int = 0
        self.hits: int = 0
        self.bank = FeatureBank()
        if det.embedding is not None:
            self.bank.update(det.embedding)

    # ----- state lifecycle -------------------------------------------------
    def activate(self, kf: KalmanFilter, frame_id: int) -> None:
        self.track_id = self.next_id()
        self.mean, self.covariance = kf.initiate(xyxy_to_xyah(self.tlbr))
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.hits = 1
        self.state = TrackState.Tracked

    def predict(self, kf: KalmanFilter) -> None:
        if self.mean is None or self.covariance is None:
            return
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.tlbr = xyah_to_xyxy(self.mean[:4])
        self.time_since_update += 1

    def update(self, det: Detection, kf: KalmanFilter, frame_id: int) -> None:
        if self.mean is None or self.covariance is None:
            self.activate(kf, frame_id)
        else:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, xyxy_to_xyah(det.tlbr)
            )
        self.tlbr = det.tlbr.astype(np.float32)
        self.score = float(det.score)
        self.cls = float(det.cls)
        self.det_ind = int(det.det_ind)
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.Tracked
        if det.embedding is not None:
            self.bank.update(det.embedding)

    def re_activate(
        self, det: Detection, kf: KalmanFilter, frame_id: int, new_track: bool = False
    ) -> None:
        if self.mean is None or self.covariance is None or new_track:
            self.mean, self.covariance = kf.initiate(xyxy_to_xyah(det.tlbr))
        else:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, xyxy_to_xyah(det.tlbr)
            )
        self.tlbr = det.tlbr.astype(np.float32)
        self.score = float(det.score)
        self.cls = float(det.cls)
        self.det_ind = int(det.det_ind)
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = TrackState.Tracked
        if det.embedding is not None:
            self.bank.update(det.embedding)


@dataclass
class DetrTracker:
    """DETR-friendly tracking backend.

    Public surface intentionally mirrors
    ``trackers.adapter.ByteTrackAdapter`` so that, when the time comes
    to integrate this backend into the pipeline, the swap is
    mechanical.

    Contract (aabb everywhere, identical to the existing backend):

      * Input:  ``dets`` is ``[N, 6]`` ``[x1, y1, x2, y2, conf, cls]``
                and ``frame`` / ``embeddings`` are optional. If
                ``embeddings`` is provided it must be ``[N, D]``
                aligned with ``dets``.
      * Output: ``[M, 8]`` ``[x1, y1, x2, y2, track_id, score, cls,
                det_ind]`` for tracks that are actively Tracked
                **and** seen this frame. ``M`` may be 0 (returns
                ``[0, 8]``).
    """

    cfg: DetrTrackerConfig
    _kf: KalmanFilter = field(init=False)
    tracked: list[STrack] = field(default_factory=list, init=False)
    lost: list[STrack] = field(default_factory=list, init=False)
    removed: list[STrack] = field(default_factory=list, init=False)
    frame_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._kf = KalmanFilter()

    # ----- public ---------------------------------------------------------
    def update(
        self,
        dets: np.ndarray,
        frame: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run one tracking step. See class docstring for the contract."""
        if dets is None or dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] != 6:
            raise ValueError(f"Expected detector output shape [N, 6], got {dets.shape}")
        if embeddings is not None:
            if embeddings.ndim != 2 or embeddings.shape[0] != dets.shape[0]:
                raise ValueError(
                    f"embeddings must be [N, D] aligned with dets, "
                    f"got {embeddings.shape} vs dets {dets.shape}"
                )
        _ = frame  # accepted for API symmetry; see class docstring.

        self.frame_id += 1
        detections = self._build_detections(dets, embeddings)
        high = [d for d in detections if d.score >= self.cfg.track_thresh]
        low = [d for d in detections if self.cfg.low_thresh <= d.score < self.cfg.track_thresh]

        pool = [t for t in self.tracked + self.lost if t.state != TrackState.Removed]
        for t in pool:
            t.predict(self._kf)

        # First stage: Tracked + Lost ↔ high-score dets (strict gate).
        matches, u_track, u_det = self._first_stage(pool, high)
        activated: list[STrack] = []
        # Tracks that were Tracked last frame but didn't get a high-score
        # match this frame — they enter the second stage, not the
        # long-term "Lost" pool yet. (BoxMOT-style.)
        just_lost: list[STrack] = []
        for ti, di in matches:
            track = pool[ti]
            if track.state == TrackState.Tracked:
                track.update(high[di], self._kf, self.frame_id)
            else:
                track.re_activate(high[di], self._kf, self.frame_id, new_track=False)
            activated.append(track)
        for ti in u_track:
            t = pool[ti]
            if t.state == TrackState.Tracked:
                t.mark_lost()
                just_lost.append(t)
            else:
                # Already-Lost tracks that didn't match high-score dets
                # stay in the Lost pool; they'll be removed by buffer
                # expiry below.
                pass

        # Second stage: refind just-lost tracks via low-score dets (loose gate).
        matches2, u_just_lost, _ = self._second_stage(just_lost, low)
        for ti, di in matches2:
            just_lost[ti].re_activate(low[di], self._kf, self.frame_id)
            activated.append(just_lost[ti])

        # Anything that fell off the second stage graduates into the
        # long-term Lost pool, where it can still be re-found on later
        # frames until the buffer expires.
        for ti in u_just_lost:
            self.lost.append(just_lost[ti])

        for di in u_det:
            track = STrack(high[di])
            track.activate(self._kf, self.frame_id)
            activated.append(track)

        live_ids = {t.track_id for t in activated}
        self.tracked = [t for t in self.tracked if t.track_id in live_ids]
        for t in activated:
            if t.track_id not in {x.track_id for x in self.tracked}:
                self.tracked.append(t)

        self.lost = [t for t in self.lost if t.track_id not in live_ids]
        kept_lost: list[STrack] = []
        for t in self.lost:
            if self.frame_id - t.frame_id > self.cfg.track_buffer:
                t.mark_removed()
                self.removed.append(t)
            elif t.state != TrackState.Removed:
                kept_lost.append(t)
        self.lost = kept_lost

        return self._emit()

    # ----- helpers --------------------------------------------------------
    def _build_detections(
        self, dets: np.ndarray, embeddings: np.ndarray | None
    ) -> list[Detection]:
        out: list[Detection] = []
        for i, d in enumerate(dets):
            tlbr = np.asarray(d[:4], dtype=np.float32)
            if (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1]) < self.cfg.min_box_area:
                continue
            emb: np.ndarray | None = None
            if embeddings is not None:
                emb = np.asarray(embeddings[i], dtype=np.float32)
            elif not self.cfg.allow_embeddingless:
                raise ValueError(
                    "DetrTracker requires embeddings when allow_embeddingless=False"
                )
            out.append(
                Detection(
                    tlbr=tlbr,
                    score=float(d[4]),
                    cls=float(d[5]),
                    det_ind=i,
                    embedding=emb,
                )
            )
        return out

    def _match_first(
        self, tracks: list[STrack], detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float32)
        det_boxes = np.asarray([d.tlbr for d in detections], dtype=np.float32)
        iou_cost = iou_distance(track_boxes, det_boxes)
        app_cost = self._appearance_cost(tracks, detections)
        cost = self._fuse(iou_cost, app_cost, tracks, detections)
        return linear_assignment(cost, self.cfg.match_thresh)

    def _match_second(
        self, tracks: list[STrack], detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float32)
        det_boxes = np.asarray([d.tlbr for d in detections], dtype=np.float32)
        iou_cost = iou_distance(track_boxes, det_boxes)
        app_cost = self._appearance_cost(tracks, detections)
        cost = self._fuse(iou_cost, app_cost, tracks, detections)
        return linear_assignment(cost, self.cfg.second_match_thresh)

    def _first_stage(
        self, pool: list[STrack], high: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match Tracked + Lost against high-score detections."""
        return self._match_first(pool, high)

    def _second_stage(
        self, lost_tracks: list[STrack], low_dets: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Refind Lost tracks via low-score detections (looser gate)."""
        return self._match_second(lost_tracks, low_dets)

    def _appearance_cost(
        self, tracks: list[STrack], detections: list[Detection]
    ) -> np.ndarray:
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        track_feats: list[np.ndarray | None] = [t.bank.mean for t in tracks]
        det_feats: list[np.ndarray | None] = [
            d.embedding.astype(np.float32) if d.embedding is not None else None
            for d in detections
        ]

        # If any side is missing, return a zero cost so the caller falls
        # back to IoU-only via :meth:`_fuse`. This is the graceful
        # degradation path.
        if any(f is None for f in track_feats) or any(f is None for f in det_feats):
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        d_dim = det_feats[0].shape[0]  # type: ignore[union-attr]
        track_mat = np.stack([pad_or_trim(f, d_dim) for f in track_feats], axis=0)  # type: ignore[arg-type]
        det_mat = np.stack([pad_or_trim(f, d_dim) for f in det_feats], axis=0)  # type: ignore[arg-type]
        return cosine_distance(track_mat, det_mat)

    def _fuse(
        self,
        iou_cost: np.ndarray,
        app_cost: np.ndarray,
        tracks: list[STrack],
        detections: list[Detection],
    ) -> np.ndarray:
        if iou_cost.size == 0:
            return iou_cost
        track_has_emb = np.asarray(
            [t.bank.mean is not None for t in tracks], dtype=bool
        )
        det_has_emb = np.asarray(
            [d.embedding is not None for d in detections], dtype=bool
        )
        cost = fuse_motion_appearance(iou_cost, app_cost, self.cfg.lambda_iou)
        # Rows / columns missing an embedding fall back to IoU-only.
        if not track_has_emb.all():
            cost[~track_has_emb, :] = iou_cost[~track_has_emb, :]
        if not det_has_emb.all():
            cost[:, ~det_has_emb] = iou_cost[:, ~det_has_emb]
        return cost

    def _emit(self) -> np.ndarray:
        rows: list[list[float]] = []
        for t in self.tracked:
            if t.state != TrackState.Tracked:
                continue
            if t.time_since_update != 0:
                continue
            if t.hits < self.cfg.min_hits:
                # BoxMOT-style: New tracks that haven't accumulated enough
                # hits are suppressed to avoid id flicker.
                continue
            rows.append(
                [
                    float(t.tlbr[0]),
                    float(t.tlbr[1]),
                    float(t.tlbr[2]),
                    float(t.tlbr[3]),
                    float(t.track_id),
                    float(t.score),
                    float(t.cls),
                    float(t.det_ind),
                ]
            )
        if not rows:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
