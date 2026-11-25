"""后端服务:启动实时引擎,通过 WebSocket 每秒向前端推送一帧数据。

启动:  python server.py          然后在浏览器打开 http://localhost:8000
接口:  GET  /            运营驾驶舱页面
       GET  /api/state   当前完整状态(JSON)
       WS   /ws          实时数据流;前端也通过它发送控制指令(调速、暂停、注入故障)
"""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from hoteltwin.engine import Engine

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
engine = Engine(CFG)
clients: set[WebSocket] = set()


async def loop():
    while True:
        await asyncio.sleep(1.0)
        frame = json.dumps(engine.tick(), ensure_ascii=False)
        for ws in list(clients):
            try:
                await ws.send_text(frame)
            except Exception:
                clients.discard(ws)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="无人酒店集群数字孪生", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/state")
def state():
    return JSONResponse(engine.init_frame())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps(engine.init_frame(), ensure_ascii=False))
    engine._last_sig = {}  # 新客户端加入后,下一拍向所有人补发完整房态
    clients.add(ws)
    try:
        while True:
            engine.control(json.loads(await ws.receive_text()))
    except WebSocketDisconnect:
        clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
