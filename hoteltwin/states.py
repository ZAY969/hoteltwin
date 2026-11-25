"""客房状态机定义:物理层和孪生层共用同一套状态与合法迁移。"""
from enum import Enum


class RoomState(str, Enum):
    VACANT_CLEAN = "VACANT_CLEAN"    # 空净
    RESERVED = "RESERVED"            # 已预订(当日到店)
    OCCUPIED = "OCCUPIED"            # 在住
    VACANT_DIRTY = "VACANT_DIRTY"    # 待清洁
    CLEANING = "CLEANING"            # 清洁中
    OUT_OF_ORDER = "OUT_OF_ORDER"    # 停售


S = RoomState

# 合法迁移表: (起始状态, 触发消息) -> 目标状态
TRANSITIONS = {
    (S.VACANT_CLEAN, "booking_assigned"): S.RESERVED,
    (S.RESERVED, "check_in"): S.OCCUPIED,
    (S.RESERVED, "no_show"): S.VACANT_CLEAN,
    (S.OCCUPIED, "check_out"): S.VACANT_DIRTY,
    (S.VACANT_DIRTY, "cleaning_start"): S.CLEANING,
    (S.CLEANING, "cleaning_end"): S.VACANT_CLEAN,
    (S.VACANT_CLEAN, "room_blocked"): S.OUT_OF_ORDER,
    (S.OUT_OF_ORDER, "room_released"): S.VACANT_CLEAN,
}

STATE_MESSAGES = {msg for (_, msg) in TRANSITIONS}
DEVICES = ["lock", "power", "network", "hvac", "plumbing", "other"]
