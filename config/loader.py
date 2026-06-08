from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from config.schema import AppConfig


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = AppConfig.model_validate(raw)
    if overrides:
        cfg = cfg.with_overrides(copy.deepcopy(overrides))
    return cfg
