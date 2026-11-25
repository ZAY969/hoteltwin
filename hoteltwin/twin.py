"""三级孪生模型:客房单元孪生 -> 单店孪生 -> 集群孪生。

孪生层只接收消息,据此维护自己的状态副本,不读取物理层的任何变量。
每间客房的孪生由三部分组成:
  1. 离散状态:客房状态机(6 个状态、8 条合法迁移)
  2. 连续状态:遥测向量(室温、湿度、CO2、功率、水流量、门锁电量、网络时延、设定温度)
  3. 推断状态:设备健康向量(由决策层根据遥测数据推断后写回)
"""
from collections import Counter, deque

import numpy as np

from .environment import TEL_VARS
from .states import DEVICES, STATE_MESSAGES, TRANSITIONS, RoomState as S

STATE_LIST = list(S)
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}


class RoomTwin:
    """客房单元孪生的离散部分:状态机、订单信息和设备健康向量。"""

    def __init__(self, hotel_id, room_id):
        self.hotel_id = hotel_id
        self.room_id = room_id
        self.state = S.VACANT_CLEAN
        self.devices = {d: True for d in DEVICES}   # 设备健康(推断值)
        self.last_change = 0.0      # 最近一次状态迁移的时刻,供超时类规则使用
        self.checkin_blocked = False
        self.guests = 0             # 在住人数(来自订单)
        self.rate = 0.0             # 当晚房价(来自订单)
        self.nights = 0
        self.t_checkout = None      # 最近一次退房时刻
        self.t_clean_start = None
        self.illegal = []           # 被拒绝的非法迁移

    def apply(self, msg) -> bool:
        kind = msg["kind"]
        if kind in STATE_MESSAGES:
            target = TRANSITIONS.get((self.state, kind))
            if target is None:
                self.illegal.append((msg["t"], self.state.value, kind))
                return False
            self.state = target
            self.last_change = msg["t"]
            if kind == "check_in":
                self.guests = msg.get("guests", 1)
                self.rate = msg.get("rate", 0.0)
                self.nights = msg.get("nights", 1)
            elif kind == "check_out":
                self.guests, self.rate, self.nights = 0, 0.0, 0
                self.t_checkout = msg["t"]
            elif kind == "cleaning_start":
                self.t_clean_start = msg["t"]
        elif kind == "device_fault":
            self.devices[msg["device"]] = False
        elif kind == "device_restored":
            self.devices[msg["device"]] = True
        elif kind == "checkin_blocked":
            self.checkin_blocked = True
        elif kind == "checkin_unblocked":
            self.checkin_blocked = False
        return True

    @property
    def faulty_devices(self):
        return [d for d, ok in self.devices.items() if not ok]


class HotelTwin:
    """单店孪生:客房孪生的集合 + 全店网络状态 + 汇总指标。"""

    def __init__(self, hotel_id, n_rooms):
        self.hotel_id = hotel_id
        self.rooms = {r: RoomTwin(hotel_id, r) for r in range(n_rooms)}
        self.net_ok = True

    def apply(self, msg) -> bool:
        if msg["kind"] == "hotel_net_down":
            self.net_ok = False
            return True
        if msg["kind"] == "hotel_net_up":
            self.net_ok = True
            return True
        return self.rooms[msg["room"]].apply(msg)

    def summary(self):
        counts = Counter(r.state for r in self.rooms.values())
        n = len(self.rooms)
        return {
            "hotel": self.hotel_id,
            "rooms": n,
            **{s.value: counts.get(s, 0) for s in S},
            "occupancy": counts.get(S.OCCUPIED, 0) / n,
            "rooms_with_fault": sum(1 for r in self.rooms.values() if r.faulty_devices),
            "net_ok": self.net_ok,
        }


class ClusterTwin:
    """集群孪生:多店汇总 + 全部客房的连续状态矩阵。"""

    def __init__(self, n_hotels, rooms_per_hotel):
        self.H, self.K = n_hotels, rooms_per_hotel
        self.N = n = n_hotels * rooms_per_hotel
        self.hotels = {h: HotelTwin(h, rooms_per_hotel) for h in range(n_hotels)}
        self.room_list = [self.hotels[h].rooms[r] for h in range(n_hotels) for r in range(rooms_per_hotel)]
        self.messages_applied = 0
        # ---- 连续状态 ----
        self.tel = np.zeros((n, len(TEL_VARS)))     # 最新遥测值
        self.tel[:, 0], self.tel[:, 1], self.tel[:, 2], self.tel[:, 5] = 27.5, 55.0, 420.0, 100.0
        self.last_seen = np.zeros(n)                # 每间客房最近一次收到数据的时刻
        self.lock_ok = np.ones(n, dtype=bool)       # 门锁自检标志
        self.other_ok = np.ones(n, dtype=bool)
        self.energy_kwh = np.zeros(n)               # 由功率测量值积分得到的累计电量
        self.water_l = np.zeros(n)                  # 由流量测量值积分得到的累计用水
        self._energy_midnight = np.zeros(n)
        self._water_midnight = np.zeros(n)
        self._day = 0
        self.readings = 0                           # 累计接收的测量值个数
        self.turnarounds = deque(maxlen=2000)       # (时刻, 退房到可售的时长, 清洁时长)

    # ---- 消息入口 ----
    def apply(self, msg) -> bool:
        self.messages_applied += 1
        if msg["kind"] == "telemetry":
            self._ingest(msg)
            return True
        ok = self.hotels[msg["hotel"]].apply(msg)
        if ok and msg["kind"] == "cleaning_end":
            rt = self.hotels[msg["hotel"]].rooms[msg["room"]]
            if rt.t_checkout is not None and rt.t_clean_start is not None:
                self.turnarounds.append((msg["t"], msg["t"] - rt.t_checkout, msg["t"] - rt.t_clean_start))
        return ok

    def _ingest(self, msg):
        t, idx, data = msg["t"], msg["idx"], msg["data"]
        day = int(t // 1440)
        if day != self._day:   # 跨天:记下零点时的累计值,用于计算"今日"用量
            self._day = day
            self._energy_midnight = self.energy_kwh.copy()
            self._water_midnight = self.water_l.copy()
        dt = np.minimum(t - self.last_seen[idx], 15.0)   # 数据中断后恢复时,最多按 15 分钟积分
        self.tel[idx] = data
        self.energy_kwh[idx] += data[:, 3] * dt / 60000.0
        self.water_l[idx] += data[:, 4] * dt
        self.last_seen[idx] = t
        self.lock_ok[idx] = msg["lock_ok"]
        self.other_ok[idx] = msg["other_ok"]
        self.readings += data.size

    def set_device(self, h, r, dev, ok):
        """决策层把推断出的设备健康状态写回孪生。"""
        if dev in DEVICES:
            self.hotels[h].rooms[r].devices[dev] = ok

    # ---- 汇总 ----
    def state_array(self):
        return np.fromiter((STATE_INDEX[r.state] for r in self.room_list), int, self.N)

    @property
    def energy_today(self):
        return self.energy_kwh - self._energy_midnight

    @property
    def water_today(self):
        return self.water_l - self._water_midnight

    def summary(self):
        per_hotel = [h.summary() for h in self.hotels.values()]
        total = sum(h["rooms"] for h in per_hotel)
        out = {"rooms": total}
        for s in S:
            out[s.value] = sum(h[s.value] for h in per_hotel)
        out["occupancy"] = out[S.OCCUPIED.value] / total
        out["rooms_with_fault"] = sum(h["rooms_with_fault"] for h in per_hotel)
        out["hotels_net_down"] = sum(1 for h in per_hotel if not h["net_ok"])
        return out

    def illegal_transitions(self):
        return [
            (h.hotel_id, r.room_id, *item)
            for h in self.hotels.values()
            for r in h.rooms.values()
            for item in r.illegal
        ]
