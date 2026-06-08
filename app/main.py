from __future__ import annotations

import os

from fastapi import FastAPI

from app.api import create_router
from app.manager import DetectionManager
from config.loader import load_config
from logging_utils.setup import setup_logging


def create_app(config_path: str | None = None) -> FastAPI:
    path = config_path or os.getenv("MOT_CONFIG", "config/default.yaml")
    cfg = load_config(path)
    setup_logging(cfg.logging.level, cfg.logging.ring_buffer_size)
    app = FastAPI(title="MOT Stream Service", version="0.1.0")
    app.include_router(create_router(DetectionManager(cfg)))
    return app


app = create_app()
