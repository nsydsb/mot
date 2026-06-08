from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from ultralytics import YOLO

from config.schema import DetectorConfig


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model_path = Path(cfg.models_dir) / f"{cfg.model_type}.pt"
        self.model = YOLO(str(self.model_path))
        # Resolve any string class names to int ids so the runtime payload
        # going into the model is always a list[int] (or None for "all").
        self.cfg.classes = self._resolve_classes(cfg.classes)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            classes=self.cfg.classes,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32, copy=False)

    def _resolve_classes(
        self, classes: Sequence[int | str] | None
    ) -> list[int] | None:
        if not classes:
            # None or empty list -> run on all classes.
            return None
        name_to_id: dict[str, int] = {
            name: idx for idx, name in self.model.names.items()
        }
        resolved: list[int] = []
        for c in classes:
            if isinstance(c, bool):
                # bool is a subclass of int; reject it explicitly to avoid
                # accidentally passing True/False as class ids.
                raise TypeError(f"classes must be int or str, got bool: {c!r}")
            if isinstance(c, int):
                if c not in self.model.names:
                    raise ValueError(
                        f"Unknown class id: {c}; model exposes ids "
                        f"0..{len(self.model.names) - 1}"
                    )
                resolved.append(c)
            elif isinstance(c, str):
                if c not in name_to_id:
                    available = sorted(name_to_id.keys())
                    raise ValueError(
                        f"Unknown class name: {c!r}; available: {available}"
                    )
                resolved.append(name_to_id[c])
            else:
                raise TypeError(
                    f"classes items must be int or str, got {type(c).__name__}"
                )
        # Preserve order, drop duplicates, keep stable sequence.
        seen: set[int] = set()
        deduped: list[int] = []
        for cid in resolved:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        return deduped
