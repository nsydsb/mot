from __future__ import annotations

import logging
from collections import deque
from threading import Lock


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 500):
        super().__init__()
        self.records: deque[str] = deque(maxlen=capacity)
        self._records_lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with self._records_lock:
            self.records.append(msg)

    def tail(self, n: int) -> list[str]:
        with self._records_lock:
            return list(self.records)[-n:]
