"""Detectors package.

Currently shipped:

* :class:`detectors.yolo_detector.YoloDetector` — the default detector,
  Ultralytics YOLO.
* :class:`detectors.detr.detr_detector.DetrDetector` — DETR / Deformable
  DETR via HuggingFace ``transformers``. Both expose the same
  ``[N, 6]`` aabb contract, so they're interchangeable at the
  pipeline boundary.
"""
