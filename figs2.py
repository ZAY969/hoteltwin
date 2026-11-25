import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("figs.py").read().split("def box")[0].split("import matplotlib.pyplot as plt")[1].replace("from matplotlib.patches import FancyBboxPatch", ""))
A = json.load(open("output/supp_agg.json"))
# 图 5 负荷扫描
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for c, col, mk in ((1, B, "o"), (2, O, "s")):
    ks = sorted(int(k.split("_k")[1].split("|")[0]) for k in A if k.startswith(f"c{c}_"))
    xs = [400 * k for k in ks]; ys = [A[f"c{c}_k{k}|P"]["p1_within_5"][0] * 100 for k in ks]
    ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.2, label=f"{c} 名运营员")
    ax.text(xs[-1] + 80, ys[-1], f"{c} 名运营员", color=INK, va="center", fontsize=8.5)
ax.axhline(95, color=SUB, lw=0.8, ls="--"); ax.text(7650, 91.8, "服务水平目标 95%", color=SUB, fontsize=8, ha="right")
ax.set_xlabel("等效客房数"); ax.set_ylabel("紧急事件 5 分钟内有人处理的比例(%)"); ax.set_ylim(70, 101); ax.set_xlim(0, 7700)
ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower left", fontsize=8.5)
fig.savefig("figs/fig5_sweep.png", bbox_inches="tight"); plt.close(fig)
# 图 6 敏感性
S = [("基准", "base", "base"), ("故障率 ×0.5", "fault_x0.5", "fault_x0.5"), ("故障率 ×2", "fault_x2", "fault_x2"),
     ("关闭自动处置", "auto_off", "base"), ("远程处理时长 ×2", "handle_10min", "base"), ("驻店发现时间 ×0.5", "base", "notice_x0.5"), ("最不利组合", "worst", "worst")][::-1]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (k, t) in zip(axs, (("detect", "故障发现时间均值(分钟)"), ("guest_hours_per_100rd", "客人受影响时长(小时/百间·天)"))):
    for i, (n, p, tr) in enumerate(S):
        a, b = A[f"{p}|P"][k][0], A[f"{tr}|T"][k][0]
        ax.plot([a, b], [i, i], color="#c9c9c9", lw=2, zorder=1)
        ax.scatter([a], [i], s=50, color=B, zorder=2, label="所提系统" if i == 0 else None, edgecolor="white", lw=1.2)
        ax.scatter([b], [i], s=50, color=O, marker="s", zorder=2, label="驻店管理" if i == 0 else None, edgecolor="white", lw=1.2)
    ax.set_yticks(range(len(S))); ax.set_yticklabels([s[0] for s in S]); ax.set_xlabel(t); ax.set_xlim(0, None)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(rect=(0, 0, 1, 0.93)); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8.5, loc="upper center", ncol=2); fig.savefig("figs/fig6_sensitivity.png", bbox_inches="tight"); plt.close(fig)
