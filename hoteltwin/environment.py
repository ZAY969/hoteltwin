"""客房环境与设备的连续量模型(物理层的一部分)。

每间客房维护一组连续的"真实值",每个采样周期按物理关系推进一步,
再叠加传感器噪声后作为遥测数据上报。孪生层只能看到带噪声的测量值。

状态量与模型(dt 为采样周期,单位分钟):
  室温 T      一阶热平衡   dT/dt = (T_out - T)/tau - q + g*n
              空调制冷量   q = clip(k*(T - T_set), 0, q_max);空调或供电故障时 q = 0
  CO2 浓度 C  质量平衡     dC/dt = G*n - lam*(C - C_out);新风失效时 lam 降低
  相对湿度 RH 一阶回归到目标值,淋浴时目标值升高
  功率 P      待机 + 空调(与 q 成正比) + 在房用电 + 清洁设备;供电故障时为 0
  水流量 F    淋浴 + 清洁用水 + 漏水(给排水故障)
  门锁电量 B  随时间线性下降,在住时更快
  网络时延 L  对数正态波动
其中 n 为在房人数,T_out 为室外温度(按日周期变化)。
"""
import math

import numpy as np

from .states import RoomState as S

# 遥测向量各列的含义
TEL_VARS = ["temp", "rh", "co2", "power", "flow", "batt", "lat", "setp"]
ST_OCC, ST_RES, ST_CLEAN = S.OCCUPIED, S.RESERVED, S.CLEANING


class RoomEnvironment:
    def __init__(self, cluster, cfg):
        self.cl = cluster
        self.p = cfg["environment"]
        c = cfg["cluster"]
        self.H, self.K = c["n_hotels"], c["rooms_per_hotel"]
        n = self.N = self.H * self.K
        self.rng = np.random.default_rng(cfg["seed"] + 11)
        self.hid = np.repeat(np.arange(self.H), self.K)
        self.rooms = [cluster.rooms[(h, r)] for h in range(self.H) for r in range(self.K)]
        p = self.p
        self.t_out_base = self.rng.normal(p["outdoor_mean_c"], p["outdoor_hotel_sd_c"], self.H)
        self.T = np.full(n, 27.5) + self.rng.normal(0, 0.5, n)
        self.RH = np.full(n, 55.0)
        self.C = np.full(n, p["co2_outdoor_ppm"], dtype=float)
        self.P = np.zeros(n)
        self.F = np.zeros(n)
        self.batt = self.rng.uniform(p["batt_init_min"], 100.0, n)
        self.lat = np.full(n, 20.0)
        self.setp = np.zeros(n)
        self.test_until = np.zeros(n)   # 主动自检:孪生下发指令后,空置客房短时开启空调
        self.test_kwh = 0.0
        self.present = np.zeros(n, dtype=bool)
        self.energy_kwh = np.zeros(n)   # 真实累计电量
        self.water_l = np.zeros(n)      # 真实累计用水
        self.online = np.ones(n, dtype=bool)

    def outdoor(self, now):
        hour = (now / 60.0) % 24
        return self.t_out_base + self.p["outdoor_amp_c"] * math.sin(2 * math.pi * (hour - 9) / 24)

    def step(self, now, dt):
        p, rng, n = self.p, self.rng, self.N
        hour = (now / 60.0) % 24
        # ---- 读取离散状态(房态、人数、设备好坏) ----
        occ = np.fromiter((r.state == ST_OCC for r in self.rooms), bool, n)
        res = np.fromiter((r.state == ST_RES for r in self.rooms), bool, n)
        cleaning = np.fromiter((r.state == ST_CLEAN for r in self.rooms), bool, n)
        guests = np.fromiter((r.guests if r.state == ST_OCC else 0 for r in self.rooms), float, n)
        pref = np.fromiter((r.pref for r in self.rooms), float, n)
        ok = {d: np.fromiter((r.devices[d] for r in self.rooms), bool, n)
              for d in ("lock", "power", "network", "hvac", "plumbing", "other")}
        net_h = np.array([self.cl.hotel_net_ok[h] for h in range(self.H)])[self.hid]

        # ---- 在房情况:夜间多在房,白天多外出 ----
        p_in = 0.95 if (hour >= 22 or hour < 8) else 0.35 if 9 <= hour < 18 else 0.7
        flip = rng.random(n) < 0.15
        self.present = np.where(flip, rng.random(n) < p_in, self.present) & occ
        n_in = guests * self.present
        awake = 7 <= hour < 23.5

        # ---- 室温:一阶热平衡 + 比例制冷 ----
        test = (now < self.test_until) & ~occ & ~res
        self.setp = np.where(occ, pref, np.where(res, p["precool_setpoint_c"], np.where(test, p.get("selftest_setpoint_c", 18.0), 0.0)))
        hvac_on = (occ | res | test) & ok["hvac"] & ok["power"]
        q = np.where(hvac_on, np.clip(p["hvac_gain_per_min"] * (self.T - self.setp), 0, p["hvac_qmax_c_per_min"]), 0.0)
        t_out = self.outdoor(now)[self.hid]
        self.T += dt * ((t_out - self.T) / p["thermal_tau_min"] - q + p["occupant_heat_c_per_min"] * n_in)
        self.T += rng.normal(0, 0.03, n)

        # ---- 功率与电量 ----
        plug = np.where(self.present & awake, rng.uniform(120, 380, n),
                        np.where(self.present, 45.0, np.where(occ, 25.0, 0.0)))
        self.P = (p["standby_w"] + q / p["hvac_qmax_c_per_min"] * p["hvac_power_w"] + plug
                  + cleaning * p["cleaning_power_w"]) * ok["power"]
        self.energy_kwh += self.P * dt / 60000.0
        self.test_kwh += float((np.where(test, q, 0.0) / p["hvac_qmax_c_per_min"] * p["hvac_power_w"] * ok["power"]).sum()) * dt / 60000.0

        # ---- 用水 ----
        sp = (0.08 if 7 <= hour < 9.5 else 0.06 if 20 <= hour < 23.5 else 0.01) if awake else 0.0
        shower = self.present & (rng.random(n) < sp)
        self.F = (shower * p["shower_lpm"] + (cleaning & (rng.random(n) < 0.4)) * p["cleaning_lpm"]
                  + (~ok["plumbing"]) * p["leak_lpm"])
        self.water_l += self.F * dt

        # ---- 湿度、CO2 ----
        self.RH += 0.3 * (55.0 + 20.0 * shower - self.RH) + rng.normal(0, 0.8, n)
        lam = np.where(ok["hvac"] & ok["power"], p["vent_rate_per_min"], p["vent_rate_fault_per_min"])
        self.C += dt * (p["co2_gen_ppm_per_min"] * n_in - lam * (self.C - p["co2_outdoor_ppm"]))

        # ---- 门锁电量、网络 ----
        self.batt = np.clip(self.batt - np.where(occ, p["batt_drain_occ"], p["batt_drain_vac"]), 0, 100)
        self.lat = 14.0 + 6.0 * rng.lognormal(0, 0.4, n)
        self.online = ok["network"] & net_h

        # ---- 传感器测量 = 真实值 + 噪声 ----
        s = p["sensor_noise"]
        meas = np.column_stack([
            self.T + rng.normal(0, s["temp"], n),
            self.RH + rng.normal(0, s["rh"], n),
            self.C + rng.normal(0, s["co2"], n),
            np.maximum(self.P * (1 + rng.normal(0, s["power_rel"], n)), 0),
            np.maximum(self.F + rng.normal(0, s["flow"], n) * (self.F > 0), 0),
            self.batt,
            self.lat,
            self.setp,
        ])
        idx = np.nonzero(self.online)[0]
        return {"t": now, "kind": "telemetry", "hotel": None, "room": None, "idx": idx,
                "data": meas[idx], "lock_ok": (ok["lock"] & (self.batt > 0))[idx], "other_ok": ok["other"][idx]}
