from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detectors.detr.postprocess import detr_outputs_to_detections


@dataclass
class DetrDetectorConfig:
    """Lightweight config for the DETR detector.

    Kept independent from ``config.schema.DetectorConfig`` on
    purpose: this backend must not touch the global schema until it's
    wired into the pipeline (which is out of scope for the current
    task).
    """

    # HF ``transformers`` model id, e.g. ``facebook/detr-resnet-50``.
    model_type: str = "facebook/detr-resnet-50"
    # Local checkpoint path. If given, takes precedence over ``model_type``.
    weights_path: str | None = None
    device: str | None = None
    conf: float = 0.5
    # Whitelist of class ids to keep. ``None`` keeps all non-background.
    classes: list[int] | None = None


class DetrDetector:
    """Thin wrapper around HuggingFace ``transformers`` DETR / Deformable DETR.

    The wrapper exists for two reasons:

    1. To keep the import surface small — the heavy ``transformers``
       / ``torch`` stack is only pulled in when this class is actually
       instantiated.
    2. To standardize output to the project's ``[N, 6]`` aabb format,
       regardless of which DETR variant the caller loads.

    The detector is **synchronous** and runs on the calling thread;
    threading concerns are handled by the pipeline layer, which is
    intentionally NOT touched here.
    """

    def __init__(self, cfg: DetrDetectorConfig) -> None:
        self.cfg = cfg
        self._image_processor = None
        self._model = None
        self._device: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._image_processor is not None:
            return
        # Imports are local so lightweight test code (e.g. the smoke
        # test that uses ``DetrTracker`` alone) doesn't require
        # torch / transformers.
        from transformers import AutoImageProcessor, AutoModel  # type: ignore

        identifier = self.cfg.weights_path or self.cfg.model_type
        self._image_processor = AutoImageProcessor.from_pretrained(identifier)
        self._model = AutoModel.from_pretrained(identifier)
        if self.cfg.device is not None:
            self._device = self.cfg.device
        else:
            try:
                import torch  # type: ignore

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        self._model.to(self._device)
        self._model.eval()

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Run a single frame and return ``[N, 6]`` xyxy+conf+cls detections."""
        if frame is None or frame.ndim != 3:
            raise ValueError(
                f"Expected HxWxC frame, got shape {None if frame is None else frame.shape}"
            )
        self._ensure_loaded()
        import torch  # type: ignore

        h, w = frame.shape[:2]
        encoding = self._image_processor(images=frame, return_tensors="pt")
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits[0].cpu().numpy()
        boxes = outputs.pred_boxes[0].cpu().numpy()
        return detr_outputs_to_detections(
            logits=logits,
            boxes=boxes,
            image_size=(w, h),
            conf_threshold=self.cfg.conf,
            target_classes=self.cfg.classes,
        )
