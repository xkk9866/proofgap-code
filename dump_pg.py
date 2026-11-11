"""Collect every number of the paper from one results directory into all_metrics.json (CPU only).

RES=/data/yaokang/ercrag/results_pg python dump_pg.py

  ret     retrieval metrics (FSR@2..6, SR, Set-EM) of every {split}.{key}.jsonl
  hops    MuSiQue FSR by hop count
  decomp  coverage kappa_k and verification error eps_k for every lattice output (decisions: pooled P, rung 0, proposal)
  boot    paired bootstrap (2,000 resamples over questions) of ProofGap minus every other method: mean difference and
          95% percentile interval, for FSR@2..5 and Set-EM
  qa      reader EM / F1 of every qa_{split}.{key}.jsonl
  stop    stopping-rule cost--quality files written by stopping.py
  latency latency.py output
Also writes the offline decision variants of the ProofGap lattice ({split}.pg_max.jsonl: max over contained
candidates instead of pooling; {split}.pg_v0.jsonl: the completeness rung of the set alone).
"""
import glob
import json
import math
import os

import numpy as np

from evaluate import gold_map, load, qa_metrics, retrieval_metrics

R = os.environ.get("RES", "/data/yaokang/ercrag/results_pg")
SPLITS = ["musique_dev", "hotpotqa_dev", "2wiki_dev", "open_musique", "open_hotpotqa", "open_2wiki"]
OURS = "pg"


def sel(r, k):
    return r["sets"].get(str(k), r["order"][:k]) if "sets" in r else r["order"][:k]


def decision_variants(split, key=OURS):
    p = f"{R}/{split}.{key}.jsonl"
    if not os.path.exists(p):
        return
    rows = load(p)
    for name in ("max", "v0"):
        with open(f"{R}/{split}.{key}_{name}.jsonl", "w") as f:
            for r in rows:
                C = r.get("cands") or {}
                V = {frozenset(x[0]): x[1] for L in C.values() for x in L}
                sets = {}
                for k, L in C.items():
                    if name == "v0":
                        b = max(L, key=lambda x: (x[1], x[2]))
                    else:
                        b = max(L, key=lambda x: (max(v for T, v in V.items() if T <= frozenset(x[0]) and len(T) <= int(k)), x[2]))
                    sets[k] = b[0]
                ks = sorted(sets, key=int)
                f.write(json.dumps({"id": r["id"], "sets": sets, "order": sets[ks[-1]] if ks else []}) + "\n")


def decomp(split, key):
    G = gold_map(split)
    rows = {r["id"]: r for r in load(f"{R}/{split}.{key}.jsonl")}
    out = {}
    for k in (2, 3, 4, 5):
        cov, fsr_p, fsr_v, fsr_c = [], [], [], []
        for q, e in G.items():
            r = rows.get(q)
            if r is None or not e["gold"] or str(k) not in r.get("cands", {}):
                continue
            g = set(e["gold"]); c = r["cands"][str(k)]
            cov.append(any(g <= set(x[0]) for x in c))
            fsr_p.append(g <= set(r["sets"][str(k)]))
            fsr_v.append(g <= set(max(c, key=lambda x: (x[1], x[2]))[0]))
            fsr_c.append(g <= set(max(c, key=lambda x: x[2])[0]))
        if cov:
            cov, fsr_p, fsr_v, fsr_c = (100 * np.array(x, float) for x in (cov, fsr_p, fsr_v, fsr_c))
            out[str(k)] = {"kappa": cov.mean(), "FSR": fsr_p.mean(), "eps_P": (cov - fsr_p).mean(),
                           "eps_V": (cov - fsr_v).mean(), "eps_chain": (cov - fsr_c).mean(), "n": len(cov)}
    return out


def boot(split, a, b, B=2000, seed=0):
    G = gold_map(split)
    ra = {r["id"]: r for r in load(f"{R}/{split}.{a}.jsonl")}
    rb = {r["id"]: r for r in load(f"{R}/{split}.{b}.jsonl")}
    ids = [q for q, e in G.items() if q in ra and q in rb and e["gold"]]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (B, len(ids)))
    res = {"n": len(ids)}
    for k in ("2", "3", "4", "5", "G"):
        def hit(r, e):
            kk = len(e["gold"]) if k == "G" else int(k)
            return float(set(e["gold"]) <= set(sel(r, kk)))
        d = np.array([hit(ra[q], G[q]) - hit(rb[q], G[q]) for q in ids])
        m = d[idx].mean(1)
        res[k] = {"diff": 100 * d.mean(), "lo": 100 * float(np.percentile(m, 2.5)), "hi": 100 * float(np.percentile(m, 97.5))}
    return res


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # keep going, report the failure in the json
        return {"error": repr(e)}


def main():
    for sp in SPLITS:
        decision_variants(sp)
    out = {"ret": {}, "hops": {}, "decomp": {}, "boot": {}, "qa": {}, "stop": {}}
    for sp in SPLITS:
        keys = []
        for p in sorted(glob.glob(f"{R}/{sp}.*.jsonl")):
            k = p[len(f"{R}/{sp}."):-6]
            if k == "gold" or k.startswith("pil_"):
                continue
            keys.append(k)
            out["ret"][f"{sp}|{k}"] = safe(retrieval_metrics, sp, p)
            r0 = load(p)[0]
            if "cands" in r0:
                out["decomp"][f"{sp}|{k}"] = safe(decomp, sp, k)
        if OURS in keys:
            for b in keys:
                if b != OURS and not b.endswith("_adapt"):
                    out["boot"][f"{sp}|{OURS}|{b}"] = safe(boot, sp, OURS, b)
        for p in sorted(glob.glob(f"{R}/qa_{sp}.*.jsonl")):
            k = p[len(f"{R}/qa_{sp}."):-6]
            out["qa"][f"{sp}|{k}"] = safe(qa_metrics, sp, p)
        if os.path.exists(f"{R}/stop_{sp}.json"):
            out["stop"][sp] = json.load(open(f"{R}/stop_{sp}.json"))
    # source hold-outs: retrieval metrics only (search-rule selection and calibration never touch evaluation splits)
    for sp in ("musique_hold", "musique_hold_open", "hotpotqa_hold", "2wiki_hold", "hotpotqa_hold_open", "2wiki_hold_open"):
        for p in sorted(glob.glob(f"{R}/{sp}.*.jsonl")):
            k = p[len(f"{R}/{sp}."):-6]
            if os.path.exists(f"/data/yaokang/ercrag/data/proc/{sp}.jsonl"):
                out["ret"][f"{sp}|{k}"] = safe(retrieval_metrics, sp, p)
    for k in [k for k in ("pg", "beam", "ce", "rerank", "pg_cc", "pg_bin", "pg_hv", "pg_4ep", "beam_4ep") if os.path.exists(f"{R}/musique_dev.{k}.jsonl")]:
        out["hops"][k] = safe(retrieval_metrics, "musique_dev", f"{R}/musique_dev.{k}.jsonl", ks=(2, 3, 4), by_hops=True)
    for f in ("latency", "depth1", "g1_selection", "seeds", "budget_pg", "budget_beam", "viewstats"):
        out[f] = json.load(open(f"{R}/{f}.json")) if os.path.exists(f"{R}/{f}.json") else {}
    json.dump(out, open(f"{R}/all_metrics.json", "w"), indent=1, default=float)
    print("ok", {k: len(v) for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
