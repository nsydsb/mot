# mot_stream_service

单路视频流实时检测追踪 Web 服务：FFmpeg 解码输入为 BGR 帧，Ultralytics YOLO 检测，简化 ByteTrack 追踪，OpenCV 渲染，FFmpeg 推送到本地 SRS RTMP。

## 安装

```bash
cd mot_stream_service
pip install -e .
```

默认配置中的 FFmpeg 路径为容器内路径 `/usr/bin/ffmpeg`。如果在宿主机直接运行，并且 FFmpeg 不在该路径，请在 YAML 配置里把 `source.ffmpeg_bin` 和 `sink.ffmpeg_bin` 改成宿主机可执行的 FFmpeg 路径。

检测模型从 `models` 目录加载。默认 `model_type` 为 `yolov8`，对应模型文件：

```text
models/yolov8.pt
```

切换模型时，把模型文件放到 `models/{model_type}.pt`，然后通过 API start 请求传入 `model_type`，或直接修改 YAML 配置。

## Docker 构建和启动

```bash
cd mot_stream_service
docker build -t mot-stream-service:latest .
docker run --rm -it --network host mot-stream-service:latest
```

如果不用 host 网络，可以映射服务端口，并把 SRS 地址改成容器可访问的地址：

```bash
docker run --rm -it -p 8000:8000 mot-stream-service:latest
```

## 启动 SRS

```bash
docker run --rm -it -p 1935:1935 -p 8080:8080 ossrs/srs:5
```

## 启动服务

容器启动推荐使用 Docker。宿主机直接运行时，先按需修改 `config/default.yaml`，例如把 `source.ffmpeg_bin` 和 `sink.ffmpeg_bin` 改成宿主机可执行的 `ffmpeg` 路径。

```bash
cd mot_stream_service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

指定其他配置文件：

```bash
MOT_CONFIG=config/default.yaml uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API 示例

```bash
curl -X POST http://127.0.0.1:9001/v1/detection/start -H "Content-Type: application/json" -d '{"source_url": "rtmp://127.0.0.1:1935/live/demostream","stream_name": "mot_demo","model_type": "WALDO30_yolov8n_416x416","overrides": {"detector.conf": 0.3,"source.fps": 15}}'
```

播放地址：

- RTMP: `rtmp://127.0.0.1/live/mot_demo`
- HTTP-FLV: `http://127.0.0.1:8080/live/mot_demo.flv`
- HLS: `http://127.0.0.1:8080/live/mot_demo.m3u8`

停止：

```bash
curl -X POST http://127.0.0.1:8000/v1/detection/stop
```

查询日志：

```bash
curl "http://127.0.0.1:8000/v1/logs?tail=100"
```

## 接口

- `POST /v1/detection/start`
- `POST /v1/detection/stop`
- `GET /v1/logs?tail=100`

第一版只允许同时运行一个检测任务。
