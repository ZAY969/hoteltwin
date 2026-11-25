"""阶段 1 运行脚本:生成一个月的数据,驱动三级孪生,并按过关标准检验。

用法:  python run_stage1.py [config.yaml]
输出:  output/messages.csv   物理层发出的全部消息(数据层)
       output/snapshots.csv  集群孪生的逐小时快照
       output/summary.json   汇总结果与过关检验
       output/cluster_week1.png  第一周集群房态图
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import simpy
import yaml

from hoteltwin.physical import DAY, VirtualCluster
from hoteltwin.twin import ClusterTwin


def run(cfg):
    env = simpy.Environment()
    c = cfg["cluster"]
    twin = ClusterTwin(c["n_hotels"], c["rooms_per_hotel"])
    messages = []

    def publish(msg):          # 数据层:记录消息并转发给孪生
        messages.append(msg)
        twin.apply(msg)

    cluster = VirtualCluster(env, cfg, publish)
    snapshots, mismatches, checks = [], 0, 0

    def snapshotter():
        nonlocal mismatches, checks
        while True:
            yield env.timeout(cfg["snapshot_minutes"])
            snapshots.append({"t_min": env.now, "day": env.now / DAY, **twin.summary()})
            # 一致性检验:逐间比较孪生状态与物理真实状态(仅用于验证,孪生本身不读取物理层)
            for (h, r), room in cluster.rooms.items():
                tw = twin.hotels[h].rooms[r]
                checks += 1
                if tw.state != room.state or tw.devices != room.devices:
                    mismatches += 1

    env.process(snapshotter())
    env.run(until=cfg["days"] * DAY)
    return messages, snapshots, twin, mismatches, checks


def main():
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = Path("output")
    out.mkdir(exist_ok=True)

    messages, snapshots, twin, mismatches, checks = run(cfg)
    msg_df = pd.DataFrame(messages)
    snap_df = pd.DataFrame(snapshots)
    msg_df.to_csv(out / "messages.csv", index=False)
    snap_df.to_csv(out / "snapshots.csv", index=False)

    n_rooms = cfg["cluster"]["n_hotels"] * cfg["cluster"]["rooms_per_hotel"]
    faults = int((msg_df["kind"] == "device_fault").sum())
    illegal = twin.illegal_transitions()
    digest = hashlib.sha256(msg_df.to_csv(index=False).encode()).hexdigest()[:16]
    summary = {
        "days": cfg["days"],
        "rooms": n_rooms,
        "messages": len(msg_df),
        "messages_by_kind": msg_df["kind"].value_counts().to_dict(),
        # 夜间入住率:每天 23:00 的在住客房占比,对应行业口径的"出租率"
        "night_occupancy": round(float(snap_df[(snap_df["t_min"] % DAY) == 23 * 60]["occupancy"].mean()), 4),
        # 全天平均在住占比:白天退房后到下一位客人入住前房间是空的,所以低于夜间入住率
        "all_day_occupied_share": round(float(snap_df["occupancy"].mean()), 4),
        "device_faults": faults,
        "faults_per_100_rooms_per_day": round(faults / n_rooms / cfg["days"] * 100, 2),
        "gate": {
            "one_month_generated": bool(snap_df["day"].max() >= cfg["days"] - 0.05),
            "illegal_transitions": len(illegal),
            "twin_physical_checks": checks,
            "twin_physical_mismatches": mismatches,
        },
        "messages_sha256_16": digest,
    }
    summary["gate"]["passed"] = (
        summary["gate"]["one_month_generated"] and not illegal and mismatches == 0
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wk = snap_df[snap_df["day"] <= 7]
        cols = ["OCCUPIED", "RESERVED", "VACANT_CLEAN", "VACANT_DIRTY", "CLEANING", "OUT_OF_ORDER"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(wk["day"], [wk[c] for c in cols], labels=cols)
        ax.set_xlabel("Day")
        ax.set_ylabel("Rooms")
        ax.set_title("Cluster twin: room states, week 1")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=6, fontsize=8, frameon=False)
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(out / "cluster_week1.png", dpi=150)
    except Exception as e:  # 图只是辅助,不影响数据
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
