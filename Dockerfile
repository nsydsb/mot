FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY config ./config
COPY source ./source
COPY detectors ./detectors
COPY trackers ./trackers
COPY render ./render
COPY sink ./sink
COPY pipeline ./pipeline
COPY logging_utils ./logging_utils
COPY models ./models
COPY README.md ./README.md

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 9010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9010"]