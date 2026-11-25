import copy, sys, numpy as np, yaml
from multiprocessing import Pool
from hoteltwin.engine import Engine
from hoteltwin.physical import DAY
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
def run(a):
    c, k, seed = a
    cfg = copy.deepcopy(CFG); cfg["seed"] = seed; cfg["live"]["warmup_days"] = 0
    cfg["operations"]["n_operators"] = c; cfg["faults"]["rate_per_room_day"] *= k
    cfg["operations"]["checkin_block_prob"] = min(cfg["operations"]["checkin_block_prob"] * k, 0.95)
    eng = Engine(cfg); eng.env.run(until=16 * DAY); t_end = eng.env.now
    lim = {1: 5, 2: 12, 3: 30}; out = {}
    for p in (1, 2, 3):
        w = [((e.t_human if e.t_human is not None else t_end) - e.t_queue) for e in eng.decision.events.values() if e.t >= 2 * DAY and e.t_queue is not None and e.priority == p]
        out[p] = (len(w), round(float(np.mean(np.array(w) <= lim[p])), 3) if w else None)
    return (c, k, seed, out)
if __name__ == "__main__":
    T = [(1, 2, 11), (1, 3, 11), (1, 3, 22), (1, 4, 11), (2, 12, 11), (2, 16, 11), (2, 16, 22)]
    with Pool(2) as p:
        for r in p.imap(run, T): print(r, flush=True)
