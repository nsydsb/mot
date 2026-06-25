# detectors/detr

DETR / Deformable DETR detector for `mot_stream_service`.

This is the **detection half** of the DETR backend. The matching
**tracking half** lives in `trackers/detr/`. Together they replace the
default `detectors/yolo_detector.py + trackers/bytetrack` pair without
changing anything in the pipeline.

## Layout

```
detectors/detr/
├── __init__.py
├── detr_detector.py     # HF transformers wrapper -> [N, 6] aabb
├── postprocess.py       # DETR logits + cxcywh -> [N, 6] aabb
└── tests/
    └── test_postprocess.py   # shape + cxcywh conversion smoke test
```

## Data contract

| Stage | Shape | Layout |
| ----- | ----- | ------ |
| detector → tracker | `[N, 6]` | `[x1, y1, x2, y2, conf, cls]`, xyxy, aabb |

## Why this is separate from `detectors/yolo_detector.py`

`yolo_detector.py` is a single-file wrapper around Ultralytics YOLO.
DETR has a fundamentally different output (set of `Q` object queries
with class logits + normalized cxcywh boxes), so the postprocess lives
in its own module (`postprocess.py`) and the wrapper is its own class
(`DetrDetector`). The two detectors share the same `[N, 6]` aabb
contract, which is what makes them drop-in interchangeable for the
tracker.

## Usage

```python
from detectors.detr import DetrDetector, DetrDetectorConfig

detector = DetrDetector(DetrDetectorConfig(
    model_type="facebook/detr-resnet-50",
    device=None,            # auto-detect cuda/cpu
    conf=0.5,
    classes=None,           # None = keep all non-background classes
))
dets = detector.predict(frame)   # -> [N, 6] float32
```

## Run the postprocess tests

```
python -m detectors.detr.tests.test_postprocess
```

No torch / no internet / no model weights required — the test builds
synthetic logits and normalized boxes.
