from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    url: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    reconnect_delay_sec: float = Field(default=2.0, ge=0.0)
    read_timeout_sec: float = Field(default=10.0, ge=0.0)
    ffmpeg_bin: str = "ffmpeg"


class DetectorConfig(BaseModel):
    model_type: str = Field(default="yolov8", pattern=r"^[A-Za-z0-9_.-]+$")
    models_dir: str = "models"
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, gt=0)
    device: str | None = None
    # Allow both class ids (int) and class names (str). String names are
    # resolved to ids inside YoloDetector once the model is loaded.
    classes: list[int | str] | None = None


class TrackerConfig(BaseModel):
    track_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1)
    min_box_area: float = Field(default=10.0, ge=0.0)


class RenderConfig(BaseModel):
    class_names: dict[int, str] | None = None
    line_width: int = Field(default=2, ge=1)


class SinkConfig(BaseModel):
    ffmpeg_bin: str = "ffmpeg"
    stream_name: str = "mot"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    bitrate: str = "2500k"
    preset: str = "veryfast"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    ring_buffer_size: int = Field(default=500, ge=1)


class StatsConfig(BaseModel):
    enabled: bool = True
    # How often the periodic stats report is emitted to the log.
    report_interval_sec: float = Field(default=60.0, gt=0)
    # How long a track is still considered "in scene" after its last sighting.
    # Prevents the count from dropping to zero on brief occlusions.
    active_window_sec: float = Field(default=5.0, gt=0)
    # Map of human-readable category name to a list of detector class ids that
    # belong to that category. Classes not present in any list are ignored in
    # the per-category tally (but still kept for diagnostic purposes).
    category_map: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "vehicle": [1, 2, 3, 5, 7],   # bicycle, car, motorcycle, bus, truck
            "person": [0],
        }
    )


class AppConfig(BaseModel):
    source: SourceConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    render: RenderConfig
    sink: SinkConfig
    logging: LoggingConfig
    stats: StatsConfig = Field(default_factory=StatsConfig)

    def with_overrides(self, overrides: dict[str, Any]) -> "AppConfig":
        data = self.model_dump()
        for key, value in overrides.items():
            target = data
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise KeyError(f"Unknown config path: {key}")
                target = target[part]
            if parts[-1] not in target:
                raise KeyError(f"Unknown config path: {key}")
            target[parts[-1]] = value
        return AppConfig.model_validate(data)
