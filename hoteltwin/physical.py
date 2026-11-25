"""虚拟物理层:用离散事件仿真代替真实的无人酒店集群。

物理层维护"真实状态",并且只通过消息(publish)向外报告变化。
孪生层和决策层看不到这里的任何变量,只能看到消息;
它们对物理层的作用只能通过下方"控制指令接口"里的几个方法下达。

autonomous=True  : 阶段 1 的行为,故障按随机时长自行修复(用于孪生模型验证)
autonomous=False : 故障不会自行恢复,必须由决策层下达指令(闭环管控)
"""
import math

import numpy as np
import simpy

from .states import DEVICES, TRANSITIONS, RoomState as S

DAY = 1440.0  # 时间单位:分钟


class PhysicalRoom:
    def __init__(self, hotel_id, room_id):
        self.hotel_id = hotel_id
        self.room_id = room_id
        self.state = S.VACANT_CLEAN
        self.devices = {d: True for d in DEVICES}  # True = 正常
        self.leave_time = None    # 在住客人的预计退房时刻
        self.pending_day = None   # 已售出但房间尚未腾出时,记录到店日的起始时刻
        self.repair_evts = {}     # 设备 -> 等待修复指令的事件
        self.block_evt = None     # 入住受阻时等待远程放行的事件
        self.expedite_evt = None  # 等待清洁时可被催办的事件
        self.guests = 0           # 在住人数
        self.pref = 24.0          # 客人设定的空调温度


class VirtualCluster:
    def __init__(self, env: simpy.Environment, cfg: dict, publish, autonomous=True):
        self.env = env
        self.cfg = cfg
        self.publish = publish  # 回调:向数据层发送一条消息
        self.autonomous = autonomous
        self.rng = np.random.default_rng(cfg["seed"])
        c = cfg["cluster"]
        self.rooms = {
            (h, r): PhysicalRoom(h, r)
            for h in range(c["n_hotels"])
            for r in range(c["rooms_per_hotel"])
        }
        self.hotel_net_ok = {h: True for h in range(c["n_hotels"])}
        self.rng2 = np.random.default_rng(cfg["seed"] + 7)   # 人数、房价等附加属性用独立随机流
        self.base_price = self.rng2.uniform(180, 320, c["n_hotels"])  # 各店基础房价(元) [占位]
        self.checkin_hook = None   # 传统模式下由前台办理入住的钩子
        self.checkins = 0
        self.lost_bookings = 0     # 因封房而流失的订单数
        self.fault_log = []        # 故障真值记录,仅用于事后评估检测效果
        self.active_faults = {}    # (店, 房, 设备) -> 故障真值记录
        self.envmodel = None
        if not autonomous:
            from .environment import RoomEnvironment
            self.envmodel = RoomEnvironment(self, cfg)
            env.process(self._telemetry_loop())
        env.process(self._daily_booking())
        for room in self.rooms.values():
            env.process(self._room_faults(room))
        for h in range(c["n_hotels"]):
            env.process(self._hotel_net(h))

    # ================= 控制指令接口(自上而下) =================
    def repair(self, hotel_id, room_id, device):
        """修复指令:远程重启成功、远程人工处理成功或现场维修完成时下达。"""
        if device == "battery":   # 更换门锁电池
            self.envmodel.batt[hotel_id * self.cfg["cluster"]["rooms_per_hotel"] + room_id] = 100.0
            return
        evt = self.rooms[(hotel_id, room_id)].repair_evts.get(device)
        if evt is not None and not evt.triggered:
            evt.succeed()

    def unblock_checkin(self, hotel_id, room_id):
        """远程放行:为入住受阻的客人远程开门或核验身份。"""
        evt = self.rooms[(hotel_id, room_id)].block_evt
        if evt is not None and not evt.triggered:
            evt.succeed()

    def expedite_cleaning(self, hotel_id, room_id):
        """催办清洁:让等待中的清洁任务立即开始。"""
        evt = self.rooms[(hotel_id, room_id)].expedite_evt
        if evt is not None and not evt.triggered:
            evt.succeed()

    def block_room(self, hotel_id, room_id):
        """封房指令:空净房发现关键设备故障时停止销售。"""
        room = self.rooms[(hotel_id, room_id)]
        if room.state == S.VACANT_CLEAN:
            self._move(room, "room_blocked")

    def release_room(self, hotel_id, room_id):
        """解封指令。"""
        room = self.rooms[(hotel_id, room_id)]
        if room.state == S.OUT_OF_ORDER:
            self._move(room, "room_released")

    def selftest(self, idx, minutes):
        """主动自检指令:让指定客房(按全局下标)的空调短时运行。"""
        self.envmodel.test_until[idx] = self.env.now + minutes

    def inject_fault(self, hotel_id, room_id, device):
        """人工注入一次故障(演示和测试用)。"""
        room = self.rooms[(hotel_id, room_id)]
        if room.devices[device]:
            self.env.process(self._one_fault(room, device))
            return True
        return False

    # ================= 消息与状态迁移 =================
    def _emit(self, kind, hotel_id, room_id=None, **extra):
        msg = {"t": self.env.now, "kind": kind, "hotel": hotel_id, "room": room_id}
        msg.update(extra)
        self.publish(msg)

    def _move(self, room, kind, **extra):
        """按状态机迁移真实状态并发消息。物理层自身也必须遵守迁移表。"""
        room.state = TRANSITIONS[(room.state, kind)]
        self._emit(kind, room.hotel_id, room.room_id, **extra)

    def _telemetry_loop(self):
        dt = float(self.cfg["environment"]["sample_minutes"])
        while True:
            yield self.env.timeout(dt)
            self.publish(self.envmodel.step(self.env.now, dt))
            for (h, r, _), rec in self.active_faults.items():   # 评估用:累计故障期间有客人在住的时长
                if self.rooms[(h, r)].state == S.OCCUPIED:
                    rec["guest_min"] = rec.get("guest_min", 0.0) + dt

    # ================= 随机数工具 =================
    def _lognormal_mean(self, mean, sigma):
        mu = math.log(mean) - sigma**2 / 2
        return float(self.rng.lognormal(mu, sigma))

    def _tri_hour(self, spec):
        lo, mode, hi = spec
        return float(self.rng.triangular(lo, mode, hi)) * 60.0

    # ================= 订单与住宿流程 =================
    def _daily_booking(self):
        d = self.cfg["demand"]
        while True:
            day_start = self.env.now
            for room in self.rooms.values():
                due_out = (
                    room.state == S.OCCUPIED
                    and room.leave_time is not None
                    and room.leave_time < day_start + DAY
                )
                if not (room.state == S.VACANT_CLEAN or due_out):
                    continue
                if self.rng.random() >= d["booking_prob"]:
                    continue
                if due_out:
                    room.pending_day = day_start  # 今日退房的房间可以再售,清洁完成后再转为已预订
                else:
                    self._move(room, "booking_assigned")
                    self.env.process(self._stay(room, day_start))
            yield self.env.timeout(DAY)

    def _stay(self, room, day_start):
        d = self.cfg["demand"]
        if self.rng.random() < d["no_show_prob"]:
            yield self.env.timeout(max(day_start + DAY - 1.0 - self.env.now, 0.0))  # 当日 23:59 释放
            self._move(room, "no_show")
            return
        arrive = day_start + self._tri_hour(d["checkin_hour"])
        yield self.env.timeout(max(arrive - self.env.now, 0.0))
        self.checkins += 1
        if self.checkin_hook is not None:      # 传统模式:到前台排队办理
            yield from self.checkin_hook(room)
        elif not self.autonomous:
            p_block = self.cfg["operations"]["checkin_block_prob"]
            if self.rng.random() < p_block:  # 自助入住受阻:证件识别失败、开不了门等
                room.block_evt = self.env.event()
                self._emit("checkin_blocked", room.hotel_id, room.room_id)
                yield room.block_evt
                room.block_evt = None
                self._emit("checkin_unblocked", room.hotel_id, room.room_id)
        if d.get("los_pmf"):   # 经验分布(公开订单数据集)
            nights = int(self.rng.choice(len(d["los_pmf"]), p=d["los_pmf"])) + 1
        else:
            nights = min(int(self.rng.geometric(1.0 / d["los_mean_nights"])), d["los_max_nights"])
        room.guests = int(self.rng2.choice([1, 2, 3], p=self.cfg["demand"].get("guests_pmf", [0.45, 0.45, 0.10])))
        room.pref = float(np.clip(self.rng2.normal(24.0, 0.8), 21.0, 27.0))
        weekend = int(day_start // DAY) % 7 in (4, 5)
        rate = round(float(self.base_price[room.hotel_id] * (1.2 if weekend else 1.0) * self.rng2.uniform(0.9, 1.1)))
        self._move(room, "check_in", guests=room.guests, nights=nights, rate=rate)
        leave = day_start + nights * DAY + self._tri_hour(d["checkout_hour"])
        room.leave_time = leave
        yield self.env.timeout(max(leave - self.env.now, 1.0))
        room.guests = 0
        self._move(room, "check_out")
        yield from self._clean(room)

    def _clean(self, room):
        c = self.cfg["cleaning"]
        delay = self._lognormal_mean(c["start_delay_mean_min"], c["start_delay_sigma"])
        room.expedite_evt = self.env.event()
        yield self.env.any_of([self.env.timeout(delay), room.expedite_evt])
        room.expedite_evt = None
        self._move(room, "cleaning_start")
        dur = max(float(self.rng.normal(c["duration_mean_min"], c["duration_sd_min"])), 5.0)
        yield self.env.timeout(dur)
        self._move(room, "cleaning_end")
        if room.pending_day is not None:  # 这间房今天已经售出:转为已预订,等待(或立即接待)客人
            day_start, room.pending_day = room.pending_day, None
            while room.state != S.VACANT_CLEAN:   # 清洁后因设备故障被封房:等待解封
                yield self.env.timeout(5)
            if self.env.now > day_start + DAY - 1.0:
                self.lost_bookings += 1          # 当天未能解封,这笔订单流失
                return
            self._move(room, "booking_assigned")
            self.env.process(self._stay(room, day_start))

    # ================= 设备故障 =================
    def _room_faults(self, room):
        f = self.cfg["faults"]
        rate_per_min = f["rate_per_room_day"] / DAY
        names = list(f["device_shares"])
        probs = np.array([f["device_shares"][n] for n in names], dtype=float)
        probs /= probs.sum()
        while True:
            yield self.env.timeout(float(self.rng.exponential(1.0 / rate_per_min)))
            dev = str(self.rng.choice(names, p=probs))
            if not room.devices[dev]:
                continue  # 该设备已处于故障中
            self.env.process(self._one_fault(room, dev))

    def _one_fault(self, room, dev):
        f = self.cfg["faults"]
        critical = f["critical_devices"]
        room.devices[dev] = False
        if self.autonomous:   # 阶段 1 的验证模式:设备自己上报故障并按随机时长自行修复
            self._emit("device_fault", room.hotel_id, room.room_id, device=dev)
            if dev in critical and room.state == S.VACANT_CLEAN:
                self._move(room, "room_blocked")
            yield self.env.timeout(self._lognormal_mean(f["repair_mean_min"], f["repair_sigma"]))
            room.devices[dev] = True
            self._emit("device_restored", room.hotel_id, room.room_id, device=dev)
            if room.state == S.OUT_OF_ORDER and all(room.devices[d] for d in critical):
                self._move(room, "room_released")
            return
        # 实时模式:故障不发任何消息,只改变设备的物理行为,要靠孪生从遥测数据中发现
        key = (room.hotel_id, room.room_id, dev)
        rec = {"h": room.hotel_id, "r": room.room_id, "dev": dev, "t0": self.env.now, "t1": None, "state0": room.state.value}
        self.fault_log.append(rec)
        self.active_faults[key] = rec
        room.repair_evts[dev] = self.env.event()
        yield room.repair_evts[dev]
        room.repair_evts.pop(dev, None)
        room.devices[dev] = True
        rec["t1"] = self.env.now
        self.active_faults.pop(key, None)

    def _hotel_net(self, h):
        f = self.cfg["faults"]
        rate_per_min = f["hotel_net_outage_per_day"] / DAY
        while True:
            yield self.env.timeout(float(self.rng.exponential(1.0 / rate_per_min)))
            self.hotel_net_ok[h] = False
            if self.autonomous:
                self._emit("hotel_net_down", h)
            yield self.env.timeout(float(self.rng.exponential(f["hotel_net_outage_mean_min"])))
            self.hotel_net_ok[h] = True
            if self.autonomous:
                self._emit("hotel_net_up", h)
