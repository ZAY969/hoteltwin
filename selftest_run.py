import copy, json
from multiprocessing import Pool
import numpy as np
import experiments as E
def task(t):
    iv, seed = t
    cfg = copy.deepcopy(E.CFG); cfg["seed"] = seed; cfg["detection"]["selftest_interval_h"] = iv; cfg["detection"]["selftest_minutes"] = 15
    r = E.run_proposed(cfg)
    h = r["_det"].get("hvac", {"n": 0, "delays": [], "false": 0})
    return {"iv": iv, "seed": seed, "hvac_n": h["n"], "hvac_delays": h["delays"], "hvac_false": h["false"], "by_state": h.get("by_state", {}), "detect": r["detect"], "detect_p90": r["detect_p90"],
            "duration": r["duration"], "guest": r["guest_hours_per_100rd"], "labor": r["labor_h_per_100rd"], "fa_day": r["false_alarms_per_day"],
            "test_kwh": r["test_kwh"], "energy_kwh": r["energy_kwh"], "n_tests": r["n_selftests"], "p1": r["p1_within_5"]}
if __name__ == "__main__":
    T = [(iv, s) for iv in (0, 24, 12, 6) for s in (11, 22, 33, 44, 55)]
    with Pool(2) as p: R = p.map(task, T)
    json.dump(R, open("output/selftest.json", "w"))
    for iv in (0, 24, 12, 6):
        rs = [r for r in R if r["iv"] == iv]; d = np.concatenate([r["hvac_delays"] for r in rs])
        print(iv, "hvac n", sum(r["hvac_n"] for r in rs), "det", len(d), "mean %.1f med %.1f p90 %.1f max %.0f" % (d.mean(), np.median(d), np.percentile(d, 90), d.max()),
              "hvac_false", sum(r["hvac_false"] for r in rs), "| detect %.1f±%.1f p90 %.1f dur %.1f guest %.2f labor %.2f fa/day %.2f" % (np.mean([r["detect"] for r in rs]), np.std([r["detect"] for r in rs], ddof=1), np.mean([r["detect_p90"] for r in rs]), np.mean([r["duration"] for r in rs]), np.mean([r["guest"] for r in rs]), np.mean([r["labor"] for r in rs]), np.mean([r["fa_day"] for r in rs])),
              "| energy +%.2f%% tests/room/day %.2f" % (100 * np.mean([r["test_kwh"] / (r["energy_kwh"] - r["test_kwh"]) for r in rs]), np.mean([r["n_tests"] for r in rs]) / 400 / 16))

    import collections
    for iv in (0, 6):
        bs = collections.defaultdict(list)
        for r in R:
            if r["iv"] == iv:
                for k, v in r["by_state"].items(): bs[k] += v
        print("iv", iv, {k: (len(v), round(float(np.mean(v)), 0), round(float(np.median(v)), 0)) for k, v in bs.items()})
