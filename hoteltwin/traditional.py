"""对照方案:传统驻店管理模式。

与本文方案共用同一个虚拟物理层(同样的订单流和故障流),区别只在"谁发现、谁处置":
  发现  没有孪生。故障靠在房客人察觉后报告、清洁人员发现、店员定时巡查发现。
  处置  每店一名前台 24 小时在岗,先到场尝试处理;处理不了再找本店工程人员。
  入住  全部由前台人工办理,需要排队。
门店之间人员不共享。
"""
import math

import numpy as np
import simpy

from .states import RoomState as S


class TraditionalOps:
    def __init__(self, env, cfg, physical):
        self.env, self.cfg, self.physical = env, cfg, physical
        self.tr = cfg["traditional"]
        self.K = cfg["cluster"]["rooms_per_hotel"]
        self.rng = np.random.default_rng(cfg["seed"] + 21)
        n_h = cfg["cluster"]["n_hotels"]
        self.staff = {h: simpy.PriorityResource(env, capacity=self.tr["frontdesk_per_hotel"]) for h in range(n_h)}
        self.checkin_waits = []
        self.staff_busy_min = 0.0
        physical.checkin_hook = self._checkin
        env.process(self._discover_loop())

    def _lognormal(self, mean, sigma=0.6):
        return float(self.rng.lognormal(math.log(mean) - sigma**2 / 2, sigma))

    def _checkin(self, room):
        t0 = self.env.now
        with self.staff[room.hotel_id].request(priority=2) as req:
            yield req
            self.checkin_waits.append(self.env.now - t0)
            dur = self._lognormal(self.tr["checkin_service_mean_min"], 0.4)
            self.staff_busy_min += dur
            yield self.env.timeout(dur)

    def _discover_loop(self):
        tr, step = self.tr, 5.0
        while True:
            yield self.env.timeout(step)
            patrol = (self.env.now % tr["patrol_interval_min"]) < step
            present = self.physical.envmodel.present
            for (h, r, dev), rec in list(self.physical.active_faults.items()):
                if "t_det" in rec:
                    continue
                room = self.physical.rooms[(h, r)]
                p = 0.0
                if room.state == S.OCCUPIED and present[h * self.K + r]:
                    p = 1 - math.exp(-step / tr["guest_notice_mean_min"][dev])
                elif room.state == S.CLEANING:
                    p = tr["cleaner_find_prob"] * (0.25 if dev == "network" else 1.0)
                if patrol:
                    p = max(p, tr["patrol_find_prob"][dev])
                if self.rng.random() < p:
                    rec["t_det"] = self.env.now
                    self.env.process(self._handle(h, r, dev, rec))

    def _handle(self, h, r, dev, rec):
        with self.staff[h].request(priority=1) as req:
            yield req
            rec["t_resp"] = self.env.now
            dur = self._lognormal(self.tr["staff_handle_mean_min"])
            self.staff_busy_min += dur
            yield self.env.timeout(dur)
        if self.rng.random() < self.tr["staff_fix_prob"][dev]:
            rec["level"] = "店员"
        else:
            hour = (self.env.now / 60.0) % 24
            mult = 1.0 if 8 <= hour < 18 else self.tr["engineer_night_factor"]
            yield self.env.timeout(self._lognormal(self.tr["engineer_mean_min"] * mult))
            rec["level"] = "工程"
        self.physical.repair(h, r, dev)
