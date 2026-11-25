import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#666666", "xtick.color": "#666666", "ytick.color": "#666666", "savefig.dpi": 300,
                     "axes.unicode_minus": False})
B, O, INK, SUB = "#257de0", "#d9730d", "#191919", "#666666"

def box(ax, x, y, w, h, text, fc="#f3f6fb", ec=B, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

def arrow(ax, p, q, text="", rad=0.0, dx=0, dy=0, color=SUB):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, connectionstyle=f"arc3,rad={rad}"))
    if text: ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha="center", va="center", fontsize=8, color=SUB,
                     bbox=dict(fc="white", ec="none", pad=1))

# 图 1 架构
fig, ax = plt.subplots(figsize=(6.5, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10.4); ax.axis("off")
L = [("应用层", "远程运营中心看板:集群指标 · 门店客房矩阵 · 遥测曲线 · 事件流"),
     ("决策层", "检测规则 · 优先级评分 · 自动处置 / 远程人工队列 / 现场工单"),
     ("孪生层", "客房孪生(状态机 + 遥测向量 + 设备健康) → 门店孪生 → 集群孪生"),
     ("连接层", "状态消息(事件触发) · 遥测消息(每 5 分钟) · 控制指令"),
     ("物理层", "客房、设备、传感器、网关与酒店管理系统(本文以虚拟集群代替)")]
for i, (n, d) in enumerate(L):
    y = 8.4 - i * 2.0
    box(ax, 1.3, y, 7.4, 1.3, "", fc="#f3f6fb" if n != "孪生层" else "#dbe9fb")
    ax.text(1.55, y + 0.95, n, fontsize=10, fontweight="bold", color=INK, va="center")
    ax.text(1.55, y + 0.42, d, fontsize=8.2, color=INK, va="center")
    if i < 4:
        arrow(ax, (3.2, y - 0.7), (3.2, y), color=B); arrow(ax, (6.8, y), (6.8, y - 0.7), color=O)
ax.text(0.55, 5.0, "状态数据(上行)", rotation=90, ha="center", va="center", color=B, fontsize=9)
ax.text(9.5, 5.0, "控制指令(下行)", rotation=90, ha="center", va="center", color=O, fontsize=9)
fig.savefig("figs/fig1_architecture.png", bbox_inches="tight"); plt.close(fig)

# 图 2 状态机
fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(-0.2, 6.2); ax.axis("off")
P = {"空净": (1.0, 2.9), "已预订": (4.2, 4.9), "在住": (7.6, 4.9), "待清洁": (7.6, 1.2), "清洁中": (4.2, 1.2), "停售": (1.0, 0.1)}
W, H = 1.5, 0.8
for n, (x, y) in P.items(): box(ax, x, y, W, H, n, fc="#dbe9fb" if n == "空净" else "#f3f6fb", fs=10)
c = lambda n: (P[n][0] + W / 2, P[n][1] + H / 2)
arrow(ax, (P["空净"][0] + W * 0.6, P["空净"][1] + H), (P["已预订"][0], P["已预订"][1] + H * 0.6), "预订分房", dy=0.42, dx=-0.5)
arrow(ax, (P["已预订"][0] + 0.25, P["已预订"][1]), (P["空净"][0] + W, P["空净"][1] + H * 0.7), "未到店", dy=-0.3, dx=0.55)
arrow(ax, (P["已预订"][0] + W, c("已预订")[1]), (P["在住"][0], c("在住")[1]), "入住", dy=0.25)
arrow(ax, (c("在住")[0], P["在住"][1]), (c("待清洁")[0], P["待清洁"][1] + H), "退房", dx=0.5)
arrow(ax, (P["待清洁"][0], c("待清洁")[1]), (P["清洁中"][0] + W, c("清洁中")[1]), "清洁开始", dy=0.25)
arrow(ax, (P["清洁中"][0], c("清洁中")[1]), (P["空净"][0] + W, P["空净"][1] + H * 0.25), "清洁结束", dy=0.32, dx=0.45)
arrow(ax, (P["空净"][0] + W * 0.25, P["空净"][1]), (P["停售"][0] + W * 0.25, P["停售"][1] + H), "客房停售", dx=-0.78)
arrow(ax, (P["停售"][0] + W * 0.6, P["停售"][1] + H), (P["空净"][0] + W * 0.6, P["空净"][1]), "客房恢复", dx=0.8)
fig.savefig("figs/fig2_fsm.png", bbox_inches="tight"); plt.close(fig)

# 图 3 检测时延
d = json.load(open("output/detection.json"))["det"]
order = [("power", "供电"), ("lock", "门锁"), ("other", "其他设备"), ("network", "网络"), ("plumbing", "给排水"), ("hvac", "空调")][::-1]
fig, ax = plt.subplots(figsize=(6.0, 2.9))
for i, (k, n) in enumerate(order):
    v = d[k]; ax.barh(i, v["median"], height=0.5, color=B)
    ax.plot([v["median"], v["p90"]], [i, i], color=INK, lw=1); ax.plot(v["p90"], i, "|", color=INK, ms=8)
    ax.text(v["p90"] * 1.15, i, f"中位数 {v['median']:.1f},90 分位 {v['p90']:.1f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels([n for _, n in order]); ax.set_xscale("log"); ax.set_xlim(1, 30000)
ax.set_xlabel("检测时延(分钟,对数坐标)"); ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
fig.savefig("figs/fig3_detection.png", bbox_inches="tight"); plt.close(fig)

# 图 4 两种模式对比
c = json.load(open("output/experiments.json"))["compare"]
M = [("detect", "故障发现时间均值(分钟)"), ("duration", "故障持续时间均值(分钟)"),
     ("guest_hours_per_100rd", "客人受影响时长(小时/百间·天)"), ("labor_h_per_100rd", "人力投入(工时/百间·天)")]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6))
for ax, (k, t) in zip(axs, M):
    p, tr = c[k]["proposed"], c[k]["traditional"]
    ax.bar([0, 1], [p[0], tr[0]], yerr=[p[1], tr[1]], width=0.55, color=[B, O], error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    for x, v in ((0, p), (1, tr)): ax.text(x, v[0] + v[1] + max(p[0], tr[0]) * 0.03, f"{v[0]:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["所提系统", "驻店管理"], fontsize=8); ax.set_title(t, fontsize=8, color=INK)
    ax.set_ylim(0, max(p[0], tr[0]) * 1.2); ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/fig4_comparison.png", bbox_inches="tight"); plt.close(fig)
