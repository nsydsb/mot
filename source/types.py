from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: np.ndarray
