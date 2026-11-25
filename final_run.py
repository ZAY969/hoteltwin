"""补充实验:负荷扫描(1/2 名运营员,全程利用率)与敏感性分析。结果逐行写入 output/supp.jsonl。"""
import copy, json, sys
from multiprocessing import Pool
import experiments as E

def task(t):
    label, mode, seed, mods = t
    cfg = copy.deepcopy(E.CFG); cfg["seed"] = seed
    for path, op, val in mods:
        d = cfg
        for p in path[:-1]: d = d[p]
        cur = d[path[-1]]
        if op == "set": d[path[-1]] = val
        elif op == "mul":
            d[path[-1]] = {k: min(v * val, 1.0) if path[-1].endswith("success") else v * val for k, v in cur.items()} if isinstance(cur, dict) else cur * val
    r = E.run_proposed(cfg) if mode == "P" else E.run_traditional(cfg)
    keep = ("n","detect","detect_p90","duration","guest_hours_per_100rd","labor_h_per_100rd","onsite_h_per_100rd",
            "p1_within_5","p1_wait_p95","hwait_p95","util_full","auto_share","false_alarms_per_day","checkin_wait","response","_det","rmse_t","rmse_p","energy_err","occ_end")
    return {"label": label, "mode": mode, "seed": seed, **{k: r.get(k) for k in keep}}

T = []
F, C, N = ("faults","rate_per_room_day"), ("operations","checkin_block_prob"), ("operations","n_operators")
for s in (11, 22, 33, 44, 55):
    T += [("base","P",s,[]), ("base","T",s,[])]
for s in (11, 22, 33):
    T += [("fault_x0.5","P",s,[(F,"mul",0.5)]), ("fault_x0.5","T",s,[(F,"mul",0.5)]),
          ("fault_x2","P",s,[(F,"mul",2)]), ("fault_x2","T",s,[(F,"mul",2)]),
          ("auto_off","P",s,[(("operations","auto_success"),"mul",0.0)]),
          ("handle_10min","P",s,[(("operations","remote_handle_mean_min"),"set",10)]),
          ("notice_x0.5","T",s,[(("traditional","guest_notice_mean_min"),"mul",0.5)])]
    T += [("worst","P",s,[(F,"mul",2),(("operations","auto_success"),"mul",0.0),(("operations","remote_handle_mean_min"),"set",10)]),
          ("worst","T",s,[(F,"mul",2),(("traditional","guest_notice_mean_min"),"mul",0.5)])]
    for k in (1, 2, 3, 4, 6, 8):
        T.append((f"c1_k{k}", "P", s, [(N,"set",1),(F,"mul",k),(C,"mul",k)]))
    for k in (4, 8, 12, 16):
        T.append((f"c2_k{k}", "P", s, [(N,"set",2),(F,"mul",k),(C,"mul",k)]))
if __name__ == "__main__":
    with Pool(2) as p, open("output/final.jsonl", "w") as f:
        for r in p.imap_unordered(task, T):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
    print("done", len(T))
