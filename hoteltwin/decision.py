"""决策层:事件识别、分级、处置分流与闭环验证。

输入只有两样:数据层转发的消息,和孪生模型的状态(房态 + 遥测矩阵)。
输出是下达给物理层的控制指令。

设备故障不会自己上报,必须从遥测数据中识别:
  断电      功率读数低于待机水平
  空调失效  室温高于设定值且没有回落趋势,持续若干个采样;或 CO2 持续偏高
  漏水      水流量持续不为零
  客房离线  超过规定时长没有收到该房数据(而同店其他客房正常)
  全店断网  全店客房同时没有数据
  门锁/其他 设备自检标志异常
  电量低    门锁电量低于阈值(预防性工单)
事件的关闭不靠"指令已发出",而靠遥测数据确实恢复正常(闭环的"验证"一步)。
"""
import math
from dataclasses import dataclass

import numpy as np
import simpy

from .states import RoomState as S

DEVICE_LABEL = {"lock": "门锁故障", "power": "客房断电", "network": "客房离线", "hvac": "空调失效",
                "plumbing": "疑似漏水", "other": "其他设备故障", "battery": "门锁电量低"}
SEVERITY = {"lock": 1.0, "power": 1.0, "plumbing": 0.9, "hvac": 0.6, "network": 0.5, "other": 0.3, "battery": 0.2}
RULE = {"lock": "自检", "other": "自检", "power": "阈值", "plumbing": "阈值持续", "hvac": "偏差趋势",
        "network": "心跳超时", "battery": "预测"}


@dataclass
class Event:
    id: int
    t: float                 # 识别时刻
    hotel: int
    room: int | None
    kind: str                # device / checkin / cleaning / conflict / net
    label: str
    rule: str
    priority: int            # 1 紧急, 2 重要, 3 一般
    score: float
    status: str = "待处置"
    level: str = ""          # 最终处置层级:自动 / 远程 / 现场
    t_response: float | None = None
    t_closed: float | None = None
    device: str | None = None
    repaired: bool = False   # 修复指令是否已下达(之后还要等遥测验证)
    evidence: str = ""       # 触发时的读数,便于人工复核
    det_delay: float | None = None   # 检测时延(评估用,对照模拟真值)
    false_alarm: bool = False        # 是否误报(评估用)
    t_queue: float | None = None     # 进入人工队列的时刻
    t_human: float | None = None     # 运营员开始处理的时刻

    def to_dict(self):
        return {"id": self.id, "t": round(self.t, 1), "h": self.hotel, "r": self.room,
                "kind": self.kind, "label": self.label, "rule": self.rule, "p": self.priority,
                "status": self.status, "level": self.level, "ev": self.evidence,
                "tr": None if self.t_response is None else round(self.t_response - self.t, 1),
                "tc": None if self.t_closed is None else round(self.t_closed - self.t, 1),
                "dd": None if self.det_delay is None else round(self.det_delay, 1), "fa": self.false_alarm}


class DecisionLayer:
    def __init__(self, env, cfg, twin, physical, truth_probe=None):
        self.env, self.cfg, self.twin, self.physical = env, cfg, twin, physical
        self.op, self.det = cfg["operations"], cfg["detection"]
        self.critical = set(cfg["faults"]["critical_devices"])
        self.truth_probe = truth_probe   # 评估用回调,真实系统中不存在
        self.rng = np.random.default_rng(cfg["seed"] + 1)
        self.operators = simpy.PriorityResource(env, capacity=self.op["n_operators"])
        self.events: dict[int, Event] = {}
        self.open_by_key: dict[tuple, Event] = {}
        self.dirty: set[int] = set()
        self.busy_minutes = []
        self._next_id = 1
        n = twin.N
        self.cnt = {k: np.zeros(n, dtype=int) for k in ("hvac", "co2", "flow")}
        self.prev_temp = twin.tel[:, 0].copy()
        self.last_test = -self.rng.uniform(0, 1, n) * self.det.get("selftest_interval_h", 0) * 60.0
        self.n_selftests = 0
        env.process(self._timeout_scanner())

    # ---------------- 工具 ----------------
    def _lognormal(self, mean, sigma):
        return float(self.rng.lognormal(math.log(mean) - sigma**2 / 2, sigma))

    def _touch(self, e):
        self.dirty.add(e.id)

    def _set(self, e, status):
        e.status = status
        if e.t_response is None and status not in ("待处置", "排队中"):
            e.t_response = self.env.now
        self._touch(e)

    def _close(self, key):
        e = self.open_by_key.pop(key, None)
        if e is None:
            return
        if e.t_response is None:
            e.t_response = self.env.now
        e.t_closed = self.env.now
        e.status = "已关闭"
        self._touch(e)
        if e.kind == "device" and e.device != "battery":
            self.twin.set_device(e.hotel, e.room, e.device, True)
            self._maybe_release(e.hotel, e.room)

    def _priority(self, severity, room_twin):
        """P = w_s*S + w_g*G + w_u*U,再映射到三个等级。"""
        w = self.op["priority_weights"]
        state = room_twin.state if room_twin is not None else None
        guest = {S.OCCUPIED: 1.0, S.RESERVED: 0.7}.get(state, 0.1)
        urgency = {S.RESERVED: 1.0, S.OCCUPIED: 0.8}.get(state, 0.2)
        score = w["severity"] * severity + w["guest"] * guest + w["urgency"] * urgency
        return (1 if score >= 0.75 else 2 if score >= 0.45 else 3), round(score, 3)

    def _new(self, key, hotel, room, kind, label, rule, priority, score, device=None, evidence=""):
        if key in self.open_by_key:
            return None
        e = Event(self._next_id, self.env.now, hotel, room, kind, label, rule, priority, score,
                  device=device, evidence=evidence)
        self._next_id += 1
        self.events[e.id] = e
        self.open_by_key[key] = e
        self._touch(e)
        if self.truth_probe and kind == "device" and device != "battery":
            self.truth_probe(e)
        if len(self.events) > 200000:
            for old in sorted(self.events)[:1000]:
                if self.events[old].t_closed is not None:
                    del self.events[old]
        return e

    # ---------------- 封房与解封 ----------------
    def _has_open_critical(self, h, r):
        return any((h, r, d) in self.open_by_key for d in self.critical)

    def _maybe_block(self, h, r):
        if self.twin.hotels[h].rooms[r].state == S.VACANT_CLEAN and self._has_open_critical(h, r):
            self.physical.block_room(h, r)

    def _maybe_release(self, h, r):
        if self.twin.hotels[h].rooms[r].state == S.OUT_OF_ORDER and not self._has_open_critical(h, r):
            self.physical.release_room(h, r)

    # ---------------- 事件识别:离散消息 ----------------
    def on_message(self, msg, accepted: bool):
        kind = msg["kind"]
        if kind == "telemetry":
            self._on_telemetry(msg["t"])
            return
        h, r = msg["hotel"], msg["room"]
        if not accepted:
            e = self._new((h, r, "conflict"), h, r, "conflict", "状态冲突", "冲突", 2, 0.5)
            if e:
                self.env.process(self._handle_review(e, (h, r, "conflict")))
        elif kind == "checkin_blocked":
            e = self._new((h, r, "checkin"), h, r, "checkin", "入住受阻", "报障", 1, 1.0)
            if e:
                self.env.process(self._handle_checkin(e))
        elif kind == "checkin_unblocked":
            self._close((h, r, "checkin"))
        elif kind == "cleaning_start":
            self._close((h, r, "cleaning"))
        elif kind in ("cleaning_end", "no_show"):
            self._maybe_block(h, r)   # 房间刚变为空净,若仍有关键设备故障则立即封房

    # ---------------- 事件识别:遥测数据 ----------------
    def _on_telemetry(self, t):
        tw, d, K = self.twin, self.det, self.twin.K
        tel = tw.tel
        age = t - tw.last_seen
        fresh = age < 1e-6
        temp, co2, power, flow, batt, setp = tel[:, 0], tel[:, 2], tel[:, 3], tel[:, 4], tel[:, 5], tel[:, 7]

        power_bad = fresh & (power < d["power_zero_w"])
        iv = d.get("selftest_interval_h", 0) * 60.0
        if iv > 0:   # 主动自检:对空净客房定期短时开启空调,让潜伏的空调故障显现
            due = fresh & (tw.state_array() == 0) & (t - self.last_test >= iv)
            if due.any():
                idx = np.nonzero(due)[0]
                self.physical.selftest(idx, d.get("selftest_minutes", 30))
                self.last_test[idx] = t
                self.n_selftests += len(idx)
        hv_now = fresh & (setp > 0) & (temp - setp > d["hvac_dev_c"]) & (temp - self.prev_temp > -0.15) & ~power_bad
        self.cnt["hvac"] = np.where(fresh, np.where(hv_now, self.cnt["hvac"] + 1, 0), self.cnt["hvac"])
        if iv > 0:   # 自检判据:空净客房在自检期间(设定值非零)功率应明显高于待机,否则判为空调失效
            hv_test = fresh & (tw.state_array() == 0) & (setp > 0) & (power < d.get("selftest_power_w", 300.0)) & ~power_bad
            self.cnt["hvac"] = np.where(hv_test, d["hvac_persist_samples"], self.cnt["hvac"])
        co2_now = fresh & (co2 > d["co2_high_ppm"]) & ~power_bad
        self.cnt["co2"] = np.where(fresh, np.where(co2_now, self.cnt["co2"] + 1, 0), self.cnt["co2"])
        self.cnt["flow"] = np.where(fresh, np.where(flow > d["flow_min_lpm"], self.cnt["flow"] + 1, 0), self.cnt["flow"])
        self.prev_temp = np.where(fresh, temp, self.prev_temp)

        stale = age > d["offline_minutes"]
        hotel_down = stale.reshape(tw.H, K).all(axis=1)
        bad = {
            "power": power_bad,
            "hvac": (self.cnt["hvac"] >= d["hvac_persist_samples"]) | (self.cnt["co2"] >= d["co2_persist_samples"]),
            "plumbing": self.cnt["flow"] >= d["flow_persist_samples"],
            "network": stale & ~np.repeat(hotel_down, K),
            "lock": fresh & ~tw.lock_ok,
            "other": fresh & ~tw.other_ok,
            "battery": fresh & (batt < d["batt_low_pct"]) & tw.lock_ok,
        }
        # 新识别的事件
        for dev, mask in bad.items():
            for idx in np.nonzero(mask)[0]:
                h, r = divmod(int(idx), K)
                if (h, r, dev) in self.open_by_key:
                    continue
                rt = tw.hotels[h].rooms[r]
                p, score = self._priority(SEVERITY[dev], rt)
                if dev == "battery":
                    p = 3
                e = self._new((h, r, dev), h, r, "device", DEVICE_LABEL[dev], RULE[dev], p, score, device=dev,
                              evidence=self._evidence(dev, idx, age))
                if e:
                    if dev != "battery":
                        tw.set_device(h, r, dev, False)
                        self._maybe_block(h, r)
                    self.env.process(self._handle_device(e))
        # 验证:修复指令下达后,读数恢复正常才关闭事件
        for key, e in list(self.open_by_key.items()):
            if e.kind != "device" or not e.repaired:
                continue
            idx = e.hotel * K + e.room
            if e.device == "network":
                normal = not stale[idx]
            else:
                normal = bool(fresh[idx]) and not bool(bad[e.device][idx])
            if normal:
                self._close(key)
        # 全店断网:靠心跳判断
        for h in range(tw.H):
            key = (h, None, "net")
            tw.hotels[h].net_ok = not bool(hotel_down[h])
            if hotel_down[h]:
                e = self._new(key, h, None, "net", "全店断网", "心跳超时", 2, 0.6,
                              evidence=f"全店 {K} 间客房均超过 {d['offline_minutes']} 分钟无数据")
                if e:
                    self._set(e, "监测中")
            elif key in self.open_by_key:
                self._close(key)

    def _evidence(self, dev, idx, age):
        tel = self.twin.tel[idx]
        if dev == "power":
            return f"功率 {tel[3]:.1f} W"
        if dev == "hvac":
            return f"室温 {tel[0]:.1f}°C,设定 {tel[7]:.1f}°C,CO2 {tel[2]:.0f} ppm"
        if dev == "plumbing":
            return f"水流量 {tel[4]:.1f} 升/分钟,已持续 {self.cnt['flow'][idx] * 5} 分钟"
        if dev == "network":
            return f"已 {age[idx]:.0f} 分钟无数据"
        if dev == "battery":
            return f"门锁电量 {tel[5]:.0f}%"
        return "设备自检标志异常"

    def _timeout_scanner(self):
        """超时类:定期扫描孪生状态,找出停留过久的客房。"""
        limit = self.op["cleaning_overdue_min"]
        while True:
            yield self.env.timeout(10)
            for h, ht in self.twin.hotels.items():
                for r, rt in ht.rooms.items():
                    if rt.state == S.VACANT_DIRTY and self.env.now - rt.last_change > limit:
                        e = self._new((h, r, "cleaning"), h, r, "cleaning", "待清洁超时", "状态超时", 3, 0.3,
                                      evidence=f"待清洁已 {self.env.now - rt.last_change:.0f} 分钟")
                        if e:
                            self.env.process(self._handle_cleaning(e))

    # ---------------- 处置分流 ----------------
    def _operator(self, e, mean_min):
        self._set(e, "排队中")
        e.t_queue = self.env.now
        with self.operators.request(priority=e.priority) as req:
            yield req
            if e.t_closed is not None:
                return False
            e.t_human = self.env.now
            self._set(e, "远程处理中")
            start = self.env.now
            yield self.env.timeout(self._lognormal(mean_min, self.op["remote_handle_sigma"]))
            self.busy_minutes.append((start, self.env.now))
        return True

    def _repair(self, e, level):
        e.level, e.repaired = level, True
        self.physical.repair(e.hotel, e.room, e.device)
        self._set(e, "待验证")

    def _handle_device(self, e):
        dev = e.device
        if dev == "battery":   # 预防性工单:直接安排现场更换
            self._set(e, "现场工单")
            yield self.env.timeout(self._lognormal(self.op["onsite_mean_min"], self.op["onsite_sigma"]))
            self._repair(e, "现场")
            return
        if self.op["auto_success"][dev] > 0:   # 第一级:自动处置
            self._set(e, "自动处置中")
            yield self.env.timeout(self.op["auto_minutes"])
            if self.rng.random() < self.op["auto_success"][dev]:
                self._repair(e, "自动")
                return
        if self.op["remote_success"][dev] > 0:   # 第二级:远程人工
            yield from self._operator(e, self.op["remote_handle_mean_min"])
            if self.rng.random() < self.op["remote_success"][dev]:
                self._repair(e, "远程")
                return
        self._set(e, "现场工单")   # 第三级:现场工单
        yield self.env.timeout(self._lognormal(self.op["onsite_mean_min"], self.op["onsite_sigma"]))
        self._repair(e, "现场")

    def _handle_checkin(self, e):
        self._set(e, "自动处置中")
        yield self.env.timeout(self.op["auto_minutes"])
        if self.rng.random() < self.op["checkin_auto_success"]:
            e.level = "自动"
        else:
            yield from self._operator(e, self.op["remote_handle_mean_min"])
            e.level = "远程"
        self.physical.unblock_checkin(e.hotel, e.room)

    def _handle_cleaning(self, e):
        ok = yield from self._operator(e, 3)
        if ok:
            e.level = "远程"
            self.physical.expedite_cleaning(e.hotel, e.room)

    def _handle_review(self, e, key):
        yield from self._operator(e, 3)
        e.level = "远程"
        self._close(key)

    # ---------------- 指标 ----------------
    def kpi(self, window=1440.0):
        now = self.env.now
        recent = [e for e in self.events.values() if e.t >= now - window]
        sla = self.op["sla_response_min"]
        open_cnt, resp, met, tot = [0, 0, 0], [[], [], []], [0, 0, 0], [0, 0, 0]
        for e in self.open_by_key.values():
            open_cnt[e.priority - 1] += 1
        levels = {"自动": 0, "远程": 0, "现场": 0}
        closed, close_t, delays, false_alarms = 0, [], [], 0
        for e in recent:
            if e.kind == "net":
                continue
            i = e.priority - 1
            if e.t_response is not None:
                dly = e.t_response - e.t
                resp[i].append(dly)
                tot[i] += 1
                met[i] += dly <= sla[e.priority]
            elif now - e.t > sla[e.priority]:
                tot[i] += 1
            if e.t_closed is not None and e.level:
                closed += 1
                levels[e.level] += 1
                close_t.append(e.t_closed - e.t)
            if e.det_delay is not None:
                delays.append(e.det_delay)
            false_alarms += e.false_alarm
        busy = sum(min(b, now) - max(a, now - window) for a, b in self.busy_minutes if b > now - window)
        self.busy_minutes = [x for x in self.busy_minutes if x[1] > now - window]
        span = min(window, now) or 1.0
        return {
            "open": open_cnt,
            "queue": len(self.operators.queue),
            "busy_ops": self.operators.count,
            "n_ops": self.op["n_operators"],
            "util": round(busy / (span * self.op["n_operators"]), 3),
            "onsite_open": sum(1 for e in self.open_by_key.values() if e.status == "现场工单"),
            "resp_avg": [round(sum(x) / len(x), 1) if x else None for x in resp],
            "sla": [round(m / t, 3) if t else None for m, t in zip(met, tot)],
            "close_avg": round(sum(close_t) / len(close_t), 1) if close_t else None,
            "events_24h": len(recent),
            "closed_24h": closed,
            "levels": levels,
            "det_delay": round(sum(delays) / len(delays), 1) if delays else None,
            "det_n": len(delays),
            "false_alarms": false_alarms,
        }
