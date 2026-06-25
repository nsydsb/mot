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
- `GET  /v1/detection/stats` — 当前 in-scene 计数的单次快照（与下面的 WS 推送数据同源）
- `GET  /v1/detection/stats/ws` — **WebSocket**，持续以 ~1Hz 推送 in-scene track 计数
- `GET  /v1/logs?tail=100`

第一版只允许同时运行一个检测任务。

## 实时统计推送（WebSocket）

> 实现方式：**服务内置的 WebSocket 端点**，不走 MQTT 也不走 SSE。前端通过 `ws://host:port/v1/detection/stats/ws` 订阅即可，无须额外 broker。

### 数据源

`TrackCategoryCounter` 维护一张 `tid → (class_id, last_seen)` 表，每帧由 inferer 线程写入，按 `active_window_sec`（默认 5s）淘汰过期记录——所以推送的"在场数"是过去几秒内**不重复的 track id 数**，能抗短暂遮挡/掉帧抖动。分类通过 `config.stats.category_map` 把 class id 映射到自定义类别名（默认 `person` / `vehicle`）。

推送节奏由独立的 `StatsPublisher` 后台线程控制（默认 1s 一次，可通过 `stats.publish_interval_sec` 调），与推理频率解耦——推理跑 30fps 也不会让前端收到 30 条/秒。

### 协议

服务端 → 客户端，每秒一条 `stats`：

```json
{
  "type": "stats",
  "task_id": "abc123",
  "stream_name": "mot_demo",
  "total": 8,
  "window_sec": 5.0,
  "categories": {"person": 3, "vehicle": 5},
  "tracked_ids": {"person": [1, 7, 9], "vehicle": [2, 4, 6, 8, 10]},
  "taken_at": 1717856123.45
}
```

空闲帧（1s 内 inferer 没新数据时）会发 `{"type": "heartbeat"}` 维持连接感。pipeline 停止时服务端主动 close。

### 限制 & 失败处理

- **没有持久化历史**：订阅从连接那一刻起开始收数据，断开期间的数据拿不到。
- **慢消费者保护**：每个订阅者有界队列（默认 8 条），满了就丢新 snapshot，不会拖累 inferer。
- **pipeline 关闭时自动 close**：客户端不需要发任何命令；服务端会推完最后一条之后干净关闭。

### 浏览器端最小示例

```html
<script>
const ws = new WebSocket("ws://" + location.host + "/v1/detection/stats/ws");
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "stats") {
    console.log("total =", msg.total, "categories =", msg.categories);
  } else if (msg.type === "heartbeat") {
    // 连接还活着，只是这 1s 没有新计数
  } else if (msg.type === "error") {
    console.error(msg.message);
  }
};
ws.onclose = () => console.log("stats stream closed");
</script>
```

### Python 客户端示例

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/v1/detection/stats/ws") as ws:
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "stats":
                print("total:", msg["total"], "cats:", msg["categories"])

asyncio.run(main())
```

### curl 不支持

curl 不能直接连 WebSocket。`/v1/detection/stats`（HTTP GET）能拿到同一份数据的**单次**快照，可以用来快速 sanity check：

```bash
curl http://127.0.0.1:8000/v1/detection/stats
# {"task_id":"...","status":"RUNNING","stream_name":"mot_demo",
#  "active_window_sec":5.0,"total":8,
#  "categories":{"person":3,"vehicle":5},
#  "tracked_ids":{"person":[1,7,9],"vehicle":[2,4,6,8,10]},
#  "taken_at":...}
```

### 配置项

```yaml
stats:
  enabled: true              # 关掉则不计数、也不推送
  publish_interval_sec: 1.0  # WebSocket 推送频率
  report_interval_sec: 60.0  # 服务端日志里打印 scene stats 的周期（与 WS 无关）
  active_window_sec: 5.0     # track 多久没出现算"离开场景"
  category_map:
    person: [0]
    vehicle: [1, 2, 3, 5, 7]
```
