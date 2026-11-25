"""批量实验:本文方案与传统驻店模式的对比,以及负荷扫描(管控幅度边界)。

用法:  python experiments.py
输出:  output/experiments.json 和屏幕上的结果表
说明:  两种方案使用相同的随机种子,因此面对的是同一套订单和故障。
       结果依赖 config.yaml 中的占位参数,只能说明机制和趋势。
"""
import copy
import json
from pathlib import Path

import numpy as np
import simpy
import yaml

from hoteltwin.engine import Engine
from hoteltwin.physical import DAY, VirtualCluster
from hoteltwin.traditional import TraditionalOps

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DAYS, WARM, SEEDS = 14, 2, [11, 22, 33, 44, 55]


def fault_stats(log, t_from):
    """从故障真值记录计算各项时长(分钟)。只统计预热期之后发生且已修复的故障。"""
    rows = [f for f in log if f["t0"] >= t_from and f["t1"] is not None and "t_det" in f]
    det = [f["t_det"] - f["t0"] for f in rows]
    dur = [f["t1"] - f["t0"] for f in rows]
    gm = [f.get("guest_min", 0.0) for f in rows]
    return {"n": len(rows), "detect": float(np.mean(det)), "detect_p90": float(np.percentile(det, 90)),
            "duration": float(np.mean(dur)), "guest_min_per_fault": float(np.mean(gm)),
            "guest_hours_per_100rd": None}


def run_proposed(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["live"]["warmup_days"] = 0
    eng = Engine(cfg)
    eng.env.run(until=(WARM + DAYS) * DAY)
    t_from, n_rooms = WARM * DAY, eng.twin.N
    st = fault_stats(eng.physical.fault_log, t_from)
    evs = [e for e in eng.decision.events.values() if e.t >= t_from]
    dev = [e for e in evs if e.kind == "device" and e.device != "battery"]
    resp = [e.t_response - e.t for e in evs if e.t_response is not None and e.kind != "net"]
    # 人工等待:进入队列到运营员开始处理的时长;到统计截止时仍在排队的,按已等待时长计入
    t_end = eng.env.now
    hw = [((e.t_human if e.t_human is not None else t_end) - e.t_queue) for e in evs if e.t_queue is not None]
    p1 = [((e.t_human if e.t_human is not None else t_end) - e.t_queue) for e in evs if e.t_queue is not None and e.priority == 1]
    onsite = [e for e in evs if e.level == "现场" and e.t_closed is not None]
    onsite_h = sum(e.t_closed - e.t for e in onsite) / 60.0
    chk = [e.t_closed - e.t for e in evs if e.kind == "checkin" and e.t_closed is not None]
    busy = sum(b - a for a, b in eng.decision.busy_minutes)
    n_ops = cfg["operations"]["n_operators"]
    rd = n_rooms * DAYS
    gm = sum(f.get("guest_min", 0.0) for f in eng.physical.fault_log if f["t0"] >= t_from)
    st.update({
        "response": float(np.mean(hw)) if hw else 0.0, "hwait_p95": float(np.percentile(hw, 95)) if hw else 0.0,
        "p1_within_5": float(np.mean(np.array(p1) <= 5.0)) if p1 else None,
        "p1_wait_p95": float(np.percentile(p1, 95)) if p1 else None,
        "checkin_wait": sum(chk) / max(eng.physical.checkins, 1),
        "false_alarms_per_day": sum(e.false_alarm for e in dev) / DAYS,
        "auto_share": float(np.mean([e.level == "自动" for e in evs if e.level])),
        "labor_h_per_100rd": (n_ops * 24 * DAYS + onsite_h) / rd * 100,
        "remote_h_per_100rd": n_ops * 24 * DAYS / rd * 100, "onsite_h_per_100rd": onsite_h / rd * 100,
        "util_last_day": eng.decision.kpi()["util"],
        "util_full": busy / (n_ops * (WARM + DAYS) * DAY),
        "guest_hours_per_100rd": gm / 60.0 / rd * 100,
        "illegal": len(eng.twin.illegal_transitions()),
    })
    det = {}
    for f in eng.physical.fault_log:
        if f["t0"] < t_from or f["t0"] > t_end - DAY: continue
        d = det.setdefault(f["dev"], {"n": 0, "delays": [], "false": 0})
        d["n"] += 1
        if "t_det" in f: d["delays"].append(f["t_det"] - f["t0"])
        if "t_det" in f: d.setdefault("by_state", {}).setdefault(f.get("state0", "?"), []).append(f["t_det"] - f["t0"])
    for e in dev:
        if e.false_alarm: det.setdefault(e.device, {"n": 0, "delays": [], "false": 0})["false"] += 1
    ev = eng._evaluation()
    occ = float(np.mean([r.state.value == "OCCUPIED" for r in eng.physical.rooms.values()]))
    st.update({"test_kwh": eng.physical.envmodel.test_kwh, "energy_kwh": float(eng.physical.envmodel.energy_kwh.sum()), "n_selftests": eng.decision.n_selftests, "_det": det, "rmse_t": ev["rmse_t"], "rmse_p": ev["rmse_p"], "energy_err": ev["energy_err"], "occ_end": occ})
    return st


def run_traditional(cfg):
    cfg = copy.deepcopy(cfg)
    env = simpy.Environment()
    phys = VirtualCluster(env, cfg, lambda m: None, autonomous=False)
    ops = TraditionalOps(env, cfg, phys)
    env.run(until=(WARM + DAYS) * DAY)
    t_from = WARM * DAY
    n_h, k = cfg["cluster"]["n_hotels"], cfg["cluster"]["rooms_per_hotel"]
    rd = n_h * k * DAYS
    st = fault_stats(phys.fault_log, t_from)
    rows = [f for f in phys.fault_log if f["t0"] >= t_from and "t_resp" in f]
    tr = cfg["traditional"]
    labor = n_h * (tr["frontdesk_per_hotel"] * 24 + tr["engineer_hours_per_day"]) * DAYS
    gm = sum(f.get("guest_min", 0.0) for f in phys.fault_log if f["t0"] >= t_from)
    st.update({
        "response": float(np.mean([f["t_resp"] - f["t_det"] for f in rows])),
        "checkin_wait": float(np.mean(ops.checkin_waits)),
        "labor_h_per_100rd": labor / rd * 100,
        "guest_hours_per_100rd": gm / 60.0 / rd * 100,
    })
    return st


def mean_sd(runs, key):
    v = [r[key] for r in runs if r.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (None, None)


def main():
    out = {}
    # ---- 实验一:两种模式对比 ----
    prop, trad = [], []
    for s in SEEDS:
        cfg = copy.deepcopy(CFG)
        cfg["seed"] = s
        prop.append(run_proposed(cfg))
        trad.append(run_traditional(cfg))
    keys = [("detect", "平均检测时延(分钟)"), ("detect_p90", "检测时延 90 分位(分钟)"),
            ("response", "发现后等待人员的时间(分钟)"), ("duration", "故障持续时间(分钟)"),
            ("guest_hours_per_100rd", "客人受影响时长(小时/百间·天)"),
            ("checkin_wait", "入住平均等待(分钟/次)"), ("labor_h_per_100rd", "人力(工时/百间·天,不含清洁)")]
    out["compare"] = {k: {"proposed": mean_sd(prop, k), "traditional": mean_sd(trad, k)} for k, _ in keys}
    print(f"\n实验一:两种模式对比({len(SEEDS)} 次重复,每次 {DAYS} 天,400 间客房)")
    print(f"{'指标':<28}{'本文方案':>18}{'传统驻店':>18}")
    for k, name in keys:
        a, b = out["compare"][k]["proposed"], out["compare"][k]["traditional"]
        print(f"{name:<28}{a[0]:>10.1f} ±{a[1]:>5.1f}{b[0]:>10.1f} ±{b[1]:>5.1f}")
    extra = {k: mean_sd(prop, k) for k in ("false_alarms_per_day", "auto_share", "remote_h_per_100rd", "onsite_h_per_100rd", "n")}
    out["proposed_extra"] = extra
    print("本文方案另:误报 %.1f 件/天,自动处置占比 %.0f%%,远程工时 %.1f,现场工时 %.1f(每百间·天)" % (
        extra["false_alarms_per_day"][0], extra["auto_share"][0] * 100, extra["remote_h_per_100rd"][0], extra["onsite_h_per_100rd"][0]))
    print("非法状态迁移:", sum(r["illegal"] for r in prop))

    # ---- 实验二:负荷扫描(1 名远程运营员,事件强度逐步放大) ----
    print("\n实验二:负荷扫描(1 名远程运营员;把故障率和入住受阻率同时放大 k 倍,相当于同样强度下客房数放大 k 倍)")
    print(f"{'k':>4}{'等效客房数':>10}{'利用率':>8}{'紧急事件5分钟内有人处理':>22}{'紧急事件等待95分位(分)':>22}")
    sweep = []
    for kx in [1, 2, 4, 8, 12, 14, 16, 18]:
        runs = []
        for s in SEEDS[:2]:
            cfg = copy.deepcopy(CFG)
            cfg["seed"] = s
            cfg["operations"]["n_operators"] = 1
            cfg["faults"]["rate_per_room_day"] *= kx
            cfg["operations"]["checkin_block_prob"] = min(cfg["operations"]["checkin_block_prob"] * kx, 0.95)
            runs.append(run_proposed(cfg))
        row = {"k": kx, "rooms_equiv": 400 * kx, "util": mean_sd(runs, "util_last_day")[0], "hwait_p95": mean_sd(runs, "hwait_p95")[0],
               "p1_within_5": mean_sd(runs, "p1_within_5")[0], "p1_wait_p95": mean_sd(runs, "p1_wait_p95")[0]}
        sweep.append(row)
        print(f"{kx:>4}{row['rooms_equiv']:>12}{row['util']:>10.1%}{row['p1_within_5']:>18.1%}{row['p1_wait_p95']:>20.1f}")
    out["sweep"] = sweep
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "experiments.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
