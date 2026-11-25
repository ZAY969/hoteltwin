"""实时引擎:把物理层、数据层、孪生层、决策层装在一起,按"拍"推进并产出前端所需的数据帧。"""
import numpy as np
import simpy

from .decision import DecisionLayer
from .physical import DAY, VirtualCluster
from .states import DEVICES, RoomState
from .twin import STATE_INDEX, ClusterTwin

OCC = STATE_INDEX[RoomState.OCCUPIED]


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg["cluster"]
        self.n_hotels, self.k = c["n_hotels"], c["rooms_per_hotel"]
        self.speed = float(cfg["live"]["speed_min_per_sec"])
        self.paused = False
        self.env = simpy.Environment()
        self.twin = ClusterTwin(self.n_hotels, self.k)
        self.state_msgs = 0
        self.physical = VirtualCluster(self.env, cfg, self._publish, autonomous=False)
        self.decision = DecisionLayer(self.env, cfg, self.twin, self.physical, truth_probe=self._truth_probe)
        self._last_sig = {}
        # 预热:先运行若干天,让集群进入正常经营状态后再对外展示
        self.env.run(until=float(cfg["live"].get("warmup_days", 0)) * DAY + 480)

    # ---------- 数据层:接收物理层消息 -> 更新孪生 -> 交给决策层 ----------
    def _publish(self, msg):
        if msg["kind"] != "telemetry":
            self.state_msgs += 1
        accepted = self.twin.apply(msg)
        self.decision.on_message(msg, accepted)

    # ---------- 评估(对照模拟真值;真实系统中做不到) ----------
    def _truth_probe(self, e):
        rec = self.physical.active_faults.get((e.hotel, e.room, e.device))
        if rec is None:
            e.false_alarm = True
        else:
            e.det_delay = e.t - rec["t0"]
            rec["t_det"], rec["event"] = e.t, e

    def _evaluation(self):
        env_m, tw = self.physical.envmodel, self.twin
        age = self.env.now - tw.last_seen
        on = age < 6.0
        rmse_t = float(np.sqrt(np.mean((tw.tel[on, 0] - env_m.T[on]) ** 2))) if on.any() else None
        rmse_p = float(np.sqrt(np.mean((tw.tel[on, 3] - env_m.P[on]) ** 2))) if on.any() else None
        true_e, twin_e = float(env_m.energy_kwh.sum()), float(tw.energy_kwh.sum())
        latent = sum(1 for key in self.physical.active_faults if key not in self.decision.open_by_key)
        return {
            "rmse_t": None if rmse_t is None else round(rmse_t, 2),
            "rmse_p": None if rmse_p is None else round(rmse_p, 1),
            "energy_err": round((twin_e - true_e) / true_e, 4) if true_e > 0 else None,
            "online": round(float(on.mean()), 3),
            "age_avg": round(float(np.minimum(age, 999).mean()), 1),
            "latent": latent,
            "active_faults": len(self.physical.active_faults),
        }

    # ---------- 帧 ----------
    def _room_sig(self, rt):
        mask = sum(1 << i for i, d in enumerate(DEVICES) if not rt.devices[d])
        return (STATE_INDEX[rt.state], mask, int(rt.checkin_blocked))

    def _rooms(self, only_changed):
        out = []
        for idx, rt in enumerate(self.twin.room_list):
            sig = self._room_sig(rt)
            if not only_changed or self._last_sig.get(idx) != sig:
                self._last_sig[idx] = sig
                out.append([idx, *sig])
        return out

    def _telemetry(self):
        """每间客房一行整数:温度x10、湿度、CO2、功率、流量x10、电量、时延、设定x10、
        数据龄(分钟)、今日电量x100(kWh)、今日用水(升)、人数、房价。"""
        tw = self.twin
        age = np.minimum(self.env.now - tw.last_seen, 999)
        guests = np.fromiter((r.guests for r in tw.room_list), float, tw.N)
        rate = np.fromiter((r.rate for r in tw.room_list), float, tw.N)
        t = tw.tel
        m = np.column_stack([t[:, 0] * 10, t[:, 1], t[:, 2], t[:, 3], t[:, 4] * 10, t[:, 5], t[:, 6], t[:, 7] * 10,
                             age, tw.energy_today * 100, tw.water_today, guests, rate])
        return np.rint(m).astype(int).tolist()

    def _kpi(self):
        tw = self.twin
        s = tw.summary()
        k = self.decision.kpi()
        st = tw.state_array()
        occ = st == OCC
        n_occ = int(occ.sum())
        tel = tw.tel
        fresh = (self.env.now - tw.last_seen) < 6.0
        oc = occ & fresh
        rate = np.fromiter((r.rate for r in tw.room_list), float, tw.N)
        now = self.env.now
        turn = [x for x in tw.turnarounds if x[0] >= now - 1440]
        comfort = float((np.abs(tel[oc, 0] - tel[oc, 7]) <= 1.5).mean()) if oc.any() else None
        k.update({
            "occ": round(s["occupancy"], 3),
            "sellable": s["rooms"] - s["OUT_OF_ORDER"],
            "ooo": s["OUT_OF_ORDER"],
            "faulty_rooms": s["rooms_with_fault"],
            "msgs": self.state_msgs,
            "readings": tw.readings,
            # 能耗与用水
            "power_kw": round(float(tel[fresh, 3].sum()) / 1000, 2),
            "energy_today": round(float(tw.energy_today.sum()), 1),
            "energy_per_occ": round(float(tw.energy_today[occ].mean()), 2) if n_occ else None,
            "flow_lpm": round(float(tel[fresh, 4].sum()), 1),
            "water_today": round(float(tw.water_today.sum()) / 1000, 2),
            "water_per_occ": round(float(tw.water_today[occ].mean()), 1) if n_occ else None,
            # 环境与舒适度
            "comfort": None if comfort is None else round(comfort, 3),
            "temp_occ": round(float(tel[oc, 0].mean()), 1) if oc.any() else None,
            "co2_occ": round(float(tel[oc, 2].mean())) if oc.any() else None,
            "co2_max": round(float(tel[fresh, 2].max())) if fresh.any() else None,
            "t_out": round(float(self.physical.envmodel.outdoor(now).mean()), 1),
            # 经营
            "adr": round(float(rate[occ].mean())) if n_occ else None,
            "revpar": round(float(rate[occ].sum()) / tw.N),
            "guests": int(sum(r.guests for r in tw.room_list)),
            "turnaround": round(sum(x[1] for x in turn) / len(turn)) if turn else None,
            "clean_time": round(sum(x[2] for x in turn) / len(turn)) if turn else None,
            # 设备
            "batt_min": round(float(tel[:, 5].min())),
            "batt_low": int((tel[:, 5] < 20).sum()),
            "lat_avg": round(float(tel[fresh, 6].mean()), 1) if fresh.any() else None,
        })
        k.update(self._evaluation())
        return k

    def init_frame(self):
        self._last_sig = {}
        evs = sorted(self.decision.events.values(), key=lambda e: e.id)[-150:]
        return {
            "type": "init", "t": self.env.now, "speed": self.speed, "paused": self.paused,
            "cfg": {"n_hotels": self.n_hotels, "k": self.k, "devices": DEVICES,
                    "states": [s.value for s in RoomState],
                    "sla": self.cfg["operations"]["sla_response_min"],
                    "sample": self.cfg["environment"]["sample_minutes"]},
            "rooms": self._rooms(False),
            "nets": [int(h.net_ok) for h in self.twin.hotels.values()],
            "tel": self._telemetry(),
            "events": [e.to_dict() for e in evs],
            "kpi": self._kpi(),
        }

    def tick(self):
        """推进一拍(现实 1 秒),返回变化量。"""
        if not self.paused:
            self.env.run(until=self.env.now + self.speed)
        dirty, self.decision.dirty = self.decision.dirty, set()
        return {
            "type": "tick", "t": self.env.now, "speed": self.speed, "paused": self.paused,
            "rooms": self._rooms(True),
            "nets": [int(h.net_ok) for h in self.twin.hotels.values()],
            "tel": self._telemetry(),
            "events": [self.decision.events[i].to_dict() for i in sorted(dirty) if i in self.decision.events],
            "kpi": self._kpi(),
        }

    # ---------- 前端控制 ----------
    def control(self, cmd):
        if "speed" in cmd:
            self.speed = max(0.5, min(float(cmd["speed"]), 120.0))
        if "paused" in cmd:
            self.paused = bool(cmd["paused"])
        if cmd.get("inject"):
            h, r = int(cmd["inject"]["h"]), int(cmd["inject"]["r"])
            dev = cmd["inject"].get("device", "lock")
            if 0 <= h < self.n_hotels and 0 <= r < self.k and dev in DEVICES:
                self.physical.inject_fault(h, r, dev)
