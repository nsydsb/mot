from __future__ import annotations

import logging
import subprocess
import time

import numpy as np

from config.schema import SourceConfig
from source.types import FramePacket

LOGGER = logging.getLogger(__name__)


class FFmpegSource:
    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen[bytes] | None = None
        self.frame_id = 0
        self.frame_size = cfg.width * cfg.height * 3

    def _cmd(self) -> list[str]:
        cmd = [
            self.cfg.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
        ]
        if self.cfg.read_timeout_sec > 0:
            cmd.extend(["-rw_timeout", str(int(self.cfg.read_timeout_sec * 1_000_000))])
        if self.cfg.url.lower().startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])
        cmd.extend([
            "-i",
            self.cfg.url,
            "-vf",
            f"fps={self.cfg.fps},scale={self.cfg.width}:{self.cfg.height}",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        return cmd

    def start(self) -> None:
        self.stop()
        LOGGER.info("Starting FFmpeg source: %s", self.cfg.url)
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )

    def _restart(self) -> None:
        LOGGER.warning("Restarting FFmpeg source after read failure")
        self.stop()
        time.sleep(self.cfg.reconnect_delay_sec)
        self.start()

    def read(self) -> FramePacket:
        if self.proc is None or self.proc.stdout is None:
            self.start()
        assert self.proc is not None and self.proc.stdout is not None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            self._restart()
            raise RuntimeError("FFmpeg source read failed")
        image = np.frombuffer(raw, dtype=np.uint8).reshape((self.cfg.height, self.cfg.width, 3))
        pkt = FramePacket(frame_id=self.frame_id, timestamp=time.time(), image=image.copy())
        self.frame_id += 1
        return pkt

    def stop(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        LOGGER.info("FFmpeg source stopped")
