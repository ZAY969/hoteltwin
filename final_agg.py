import json, collections, numpy as np
R = [json.loads(l) for l in open("output/final.jsonl")]
G = collections.defaultdict(list)
for r in R: G[(r["label"], r["mode"])].append(r)
def m(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0] if v else None
K = ("n","detect","detect_p90","duration","guest_hours_per_100rd","labor_h_per_100rd","onsite_h_per_100rd","p1_within_5","p1_wait_p95",
     "hwait_p95","util_full","auto_share","false_alarms_per_day","checkin_wait","response","rmse_t","rmse_p","energy_err","occ_end")
A = {f"{l}|{mo}": {"runs": len(rs), **{k: m(rs, k) for k in K}} for (l, mo), rs in sorted(G.items())}
det = {}
for r in G[("base", "P")]:
    for dev, d in r["_det"].items():
        x = det.setdefault(dev, {"n": 0, "delays": [], "false": 0}); x["n"] += d["n"]; x["delays"] += d["delays"]; x["false"] += d["false"]
D = {dev: {"faults": d["n"], "detected": len(d["delays"]), "mean": round(float(np.mean(d["delays"])), 1), "median": round(float(np.median(d["delays"])), 1),
           "p90": round(float(np.percentile(d["delays"], 90)), 1), "max": round(float(np.max(d["delays"])), 1), "false": d["false"]} for dev, d in det.items()}
json.dump({"agg": A, "det": D}, open("output/final_agg.json", "w"), ensure_ascii=False, indent=1)
json.dump({"det": D}, open("output/detection.json", "w"), ensure_ascii=False)
json.dump({"compare": {k: {"proposed": A["base|P"][k], "traditional": A["base|T"][k]} for k in ("detect","detect_p90","duration","guest_hours_per_100rd","labor_h_per_100rd","checkin_wait")}}, open("output/experiments.json", "w"))
json.dump({k: v for k, v in A.items()}, open("output/supp_agg.json", "w"), ensure_ascii=False)
for k, v in A.items():
    show = ("util_full","p1_within_5","p1_wait_p95") if k.startswith("c") else ("detect","detect_p90","duration","guest_hours_per_100rd","labor_h_per_100rd","onsite_h_per_100rd","auto_share","false_alarms_per_day","checkin_wait","p1_within_5","occ_end")
    print(k, v["runs"], {s: v[s] for s in show if v[s] is not None})
for dev, d in D.items(): print(dev, d)
print("fid", A["base|P"]["rmse_t"], A["base|P"]["rmse_p"], A["base|P"]["energy_err"], "total", sum(d["faults"] for d in D.values()), sum(d["detected"] for d in D.values()), sum(d["false"] for d in D.values()))
