"""阶段 1 过关检验。运行:  python -m pytest tests -q"""
import copy
from pathlib import Path

import yaml

from hoteltwin.states import RoomState as S
from hoteltwin.twin import RoomTwin
from run_stage1 import run

CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8"))


def small_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["days"] = 10
    cfg["cluster"] = {"n_hotels": 3, "rooms_per_hotel": 20}
    return cfg


def test_twin_rejects_illegal_transition():
    room = RoomTwin(0, 0)
    assert not room.apply({"t": 1.0, "kind": "check_out", "hotel": 0, "room": 0})
    assert room.state == S.VACANT_CLEAN and len(room.illegal) == 1


def test_twin_follows_a_normal_stay():
    room = RoomTwin(0, 0)
    for k in ["booking_assigned", "check_in", "check_out", "cleaning_start", "cleaning_end"]:
        assert room.apply({"t": 0.0, "kind": k, "hotel": 0, "room": 0})
    assert room.state == S.VACANT_CLEAN and not room.illegal


def test_no_illegal_transitions_and_twin_matches_physical():
    _, _, twin, mismatches, checks = run(small_cfg())
    assert twin.illegal_transitions() == []
    assert checks > 0 and mismatches == 0


def test_same_seed_gives_same_result():
    a, *_ = run(small_cfg())
    b, *_ = run(small_cfg())
    assert a == b


def test_twin_detects_a_lost_message():
    """丢掉一条消息后,孪生应当把后续消息识别为非法迁移,而不是悄悄出错。"""
    room = RoomTwin(0, 0)
    room.apply({"t": 0.0, "kind": "booking_assigned", "hotel": 0, "room": 0})
    # check_in 消息丢失
    assert not room.apply({"t": 9.0, "kind": "check_out", "hotel": 0, "room": 0})
    assert len(room.illegal) == 1


def _live_engine(ticks=720):
    from hoteltwin.engine import Engine

    cfg = copy.deepcopy(CFG)
    cfg["live"]["warmup_days"] = 0
    eng = Engine(cfg)
    for _ in range(ticks):
        eng.tick()
    return eng


def test_live_engine_closes_the_loop():
    """实时系统:运行 5 天后,房态与物理层一致,没有非法迁移,绝大多数事件已闭环。"""
    eng = _live_engine()
    assert eng.twin.illegal_transitions() == []
    for (h, r), room in eng.physical.rooms.items():
        assert eng.twin.hotels[h].rooms[r].state == room.state
    evs = list(eng.decision.events.values())
    closed = [e for e in evs if e.t_closed is not None]
    assert len(evs) > 50 and len(closed) / len(evs) > 0.9


def test_faults_are_detected_from_telemetry():
    """故障不自行上报:断电应在一个采样周期内被识别,漏水应在约 30 分钟内被识别。"""
    eng = _live_engine()
    evs = list(eng.decision.events.values())
    power = [e.det_delay for e in evs if e.device == "power" and e.det_delay is not None]
    leak = [e.det_delay for e in evs if e.device == "plumbing" and e.det_delay is not None]
    assert power and max(power) <= 5.0
    assert leak and max(leak) <= 35.0
    # 已修复的故障都被检测到过(修复只能由检测触发)
    assert all(f["t1"] is None or f["t1"] > f["t0"] for f in eng.physical.fault_log)


def test_twin_fidelity():
    """孪生的连续状态与物理真值的偏差应与传感器噪声同量级,电量积分误差应很小。"""
    k = _live_engine(300)._kpi()
    assert k["rmse_t"] < 0.4
    assert abs(k["energy_err"]) < 0.01
