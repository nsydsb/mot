from __future__ import annotations

import time

import cv2
import numpy as np

from config.schema import RenderConfig


class Renderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self.last_time: float | None = None
        self.fps = 0.0

    def render(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        now = time.time()
        if self.last_time is not None:
            dt = max(now - self.last_time, 1e-6)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self.last_time = now

        out = frame.copy()
        for trk in tracks:
            x1, y1, x2, y2, tid, _conf, _cls, _age = trk
            color = self._color(int(tid))
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, color, self.cfg.line_width)
            label = f"#{int(tid)}"
            cv2.putText(out, label, (p1[0], max(20, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(out, f"FPS {self.fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2)
        return out

    def _class_name(self, cls: int) -> str:
        if self.cfg.class_names and cls in self.cfg.class_names:
            return self.cfg.class_names[cls]
        return str(cls)

    @staticmethod
    def _color(track_id: int) -> tuple[int, int, int]:
        return ((37 * track_id) % 255, (17 * track_id + 99) % 255, (29 * track_id + 173) % 255)
