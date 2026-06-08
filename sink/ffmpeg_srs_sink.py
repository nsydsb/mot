from __future__ import annotations

import logging
import subprocess

import numpy as np

from config.schema import SinkConfig

LOGGER = logging.getLogger(__name__)


class FFmpegSrsSink:
    def __init__(self, cfg: SinkConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen[bytes] | None = None

    @property
    def rtmp_url(self) -> str:
        return f"rtmp://127.0.0.1/live/{self.cfg.stream_name}_ai"

    @property
    def play_urls(self) -> dict[str, str]:
        return {
            "rtmp": self.rtmp_url,
            "http_flv": f"http://127.0.0.1:8080/live/{self.cfg.stream_name}_ai.flv",
            "hls": f"http://127.0.0.1:8080/live/{self.cfg.stream_name}_ai.m3u8",
        }

    def start(self) -> None:
        self.stop()
        cmd = [
            self.cfg.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.cfg.width}x{self.cfg.height}",
            "-r",
            str(self.cfg.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.cfg.preset,
            "-tune",
            "zerolatency",
            "-b:v",
            self.cfg.bitrate,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "flv",
            self.rtmp_url,
        ]
        LOGGER.info("Starting FFmpeg sink: %s", self.rtmp_url)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            self.start()
        assert self.proc is not None and self.proc.stdin is not None
        if self.proc.poll() is not None:
            raise RuntimeError("FFmpeg sink exited")
        self.proc.stdin.write(frame.tobytes())

    def stop(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        LOGGER.info("FFmpeg sink stopped")
