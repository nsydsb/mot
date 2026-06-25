# trackers/detr

DETR-friendly tracking backend for `mot_stream_service`.

This is the **tracking half** of the DETR backend. The matching
**detection half** lives in `detectors/detr/`. Together they replace
the default `detectors/yolo_detector.py + trackers/bytetrack` pair
without changing anything in the pipeline.

## Layout

```
trackers/
├── common/                       # shared by every tracker in the project
│   ├── __init__.py
│   ├── basetrack.py              # BaseTrack + TrackState
│   ├── detection.py              # Detection dataclass + xyxy<->xyah
│   ├── kalman_filter.py          # xyah constant-velocity KF
│   ├── iou.py                    # bbox_ious, iou_distance
│   └── matching.py               # gated Hungarian assignment
├── bytetrack/                    # existing motion-only tracker
│   ├── basetrack.py              # back-compat shim -> trackers.common
│   ├── iou.py                    # back-compat shim
│   ├── kalman_filter.py          # back-compat shim
│   ├── matching.py               # back-compat shim
│   └── bytetrack.py              # BYTETracker (IoU only)
└── detr/                         # this package: motion + appearance
    ├── __init__.py
    ├── detr_tracker.py           # DetrTracker (BoxMOT-style core)
    ├── appearance.py             # FeatureBank + cosine distance
    ├── appearance_matching.py    # motion/appearance cost fusion
    ├── demo.py                   # offline 4-frame demo (no torch)
    └── tests/
        └── test_smoke.py         # 6 contract smoke tests
```

## Data contract

| Stage | Shape | Layout |
| ----- | ----- | ------ |
| detector → tracker | `[N, 6]` | `[x1, y1, x2, y2, conf, cls]`, xyxy, aabb |
| tracker → caller   | `[M, 8]` | `[x1, y1, x2, y2, track_id, score, cls, det_ind]`, xyxy, aabb |

Empty inputs / no live tracks return `[0, 6]` / `[0, 8]` of `float32`
without raising.

## Why this is separate from `trackers/bytetrack`

`trackers/bytetrack` is pure motion (IoU-only matching). DETR /
transformer-style backends benefit a lot from an appearance term on
top of motion, so `trackers/detr` adds:

* a per-track `FeatureBank` (EMA + bounded ring) — `appearance.py`
* a fused motion+appearance cost matrix — `appearance_matching.py`
* a second matching stage that refinds just-lost tracks via
  low-score detections — `detr_tracker.py`

The two trackers share everything else (BaseTrack, KalmanFilter, IoU,
Hungarian matching) through `trackers/common/`. New trackers should
import from `trackers.common`, not from each other.

## Usage

```python
from trackers.detr import DetrTracker, DetrTrackerConfig

tracker = DetrTracker(DetrTrackerConfig(
    track_thresh=0.5,
    match_thresh=0.8,
    second_match_thresh=0.6,
    track_buffer=30,
    lambda_iou=0.5,                # motion/appearance weight
    min_hits=3,                    # suppress id flicker on New tracks
))
dets = ...                         # [N, 6] from any detector
embs = ...                         # optional [N, D] appearance embeddings
tracks = tracker.update(dets, embeddings=embs)   # [M, 8]
```

`embeddings` is optional. When omitted (and `allow_embeddingless` is
left at its default `True`), the tracker falls back to IoU-only
matching for that frame.

## Run the smoke tests

```
python -m trackers.detr.tests.test_smoke
```

Six cases: importable without torch, empty input, id stability across
frames, second-stage refind, aabb geometry, `allow_embeddingless=False`
strict path.

## Run the offline demo

```
python -m trackers.detr.demo
```

Synthesizes a random frame, fabricates `[N, 6]` detections + random
embeddings, prints ids across 4 frames.

## Integrating into the pipeline later

The integration is a one-line swap in `pipeline/pipeline.py`:

```python
- from trackers.adapter import ByteTrackAdapter
- self.tracker = ByteTrackAdapter(self.cfg.tracker)
+ from trackers.detr import DetrTracker, DetrTrackerConfig
+ self.tracker = DetrTracker(DetrTrackerConfig(
+     track_thresh=self.cfg.tracker.track_thresh,
+     low_thresh=self.cfg.tracker.low_thresh,
+     track_buffer=self.cfg.tracker.track_buffer,
+     match_thresh=self.cfg.tracker.match_thresh,
+     min_hits=3,
+ ))
```

The `update(dets, frame)` call site already matches the contract
this tracker exposes, so the worker / sink / render layers do not
need to change.
