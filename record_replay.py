"""录制一段后端数据帧,生成不需要后端也能播放的演示页面 output/replay.html。

用法:  python record_replay.py [帧数,默认 300]
"""
import json
import sys
from pathlib import Path

import yaml

from hoteltwin.engine import Engine

root = Path(__file__).parent
cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
engine = Engine(cfg)
frames = [engine.init_frame()] + [engine.tick() for _ in range(n)]
html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
payload = "<script>window.REPLAY_FRAMES=" + json.dumps(frames, ensure_ascii=False, separators=(",", ":")) + ";</script>\n<script>"
out = root / "output" / "replay.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html.replace("<script>", payload, 1), encoding="utf-8")
print(out, round(out.stat().st_size / 1024), "KB", len(frames), "frames")

# 另存一份"页面 + 数据文件"的版本,供网页托管使用(数据较大,不宜内嵌)
site = root / "output" / "replay_site"
site.mkdir(exist_ok=True)
(site / "frames.json").write_text("[\n" + ",\n".join(json.dumps(f, ensure_ascii=False, separators=(",", ":")) for f in frames) + "\n]", encoding="utf-8")
(site / "index.html").write_text(html.replace("<script>", "<script>window.REPLAY_URL='frames.json';</script>\n<script>", 1), encoding="utf-8")
print(site)
