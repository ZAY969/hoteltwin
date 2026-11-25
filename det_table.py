"""分设备检测性能:5 个种子 x 14 天(预热 2 天)。"""
import copy, json
from pathlib import Path
import numpy as np, yaml
from hoteltwin.engine import Engine
from hoteltwin.physical import DAY
CFG = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
DAYS, WARM, SEEDS = 14, 2, [11, 22, 33, 44, 55]
agg = {}
fid = []
for s in SEEDS:
    cfg = copy.deepcopy(CFG); cfg["seed"] = s; cfg["live"]["warmup_days"] = 0
    eng = Engine(cfg); eng.env.run(until=(WARM + DAYS) * DAY)
    t0, t_end = WARM * DAY, eng.env.now
    ev = eng._evaluation(); fid.append((ev["rmse_t"], ev["rmse_p"], ev["energy_err"]))
    for f in eng.physical.fault_log:
        if f["t0"] < t0 or f["t0"] > t_end - DAY: continue   # 最后一天发生的故障不计,避免截断
        d = agg.setdefault(f["dev"], {"n": 0, "det": [], "fa": 0})
        d["n"] += 1
        if "t_det" in f: d["det"].append(f["t_det"] - f["t0"])
    for e in eng.decision.events.values():
        if e.t >= t0 and e.kind == "device" and e.false_alarm:
            agg.setdefault(e.device, {"n": 0, "det": [], "fa": 0})["fa"] += 1
out = {}
for dev, d in agg.items():
    a = np.array(d["det"]) if d["det"] else np.array([np.nan])
    out[dev] = {"faults": d["n"], "detected": len(d["det"]), "mean": round(float(np.nanmean(a)), 1),
                "median": round(float(np.nanmedian(a)), 1), "p90": round(float(np.nanpercentile(a, 90)), 1),
                "max": round(float(np.nanmax(a)), 1), "false": d["fa"]}
    print(dev, out[dev])
print("fidelity", fid)
print(eng.physical.fault_log[0].keys())
json.dump({"det": out, "fid": fid}, open("output/detection.json", "w"), ensure_ascii=False, indent=1)
