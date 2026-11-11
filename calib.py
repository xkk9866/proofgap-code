"""Aggregation rules and calibration of the completeness estimate (CPU only).

For each split and budget k, from the stored lattice candidates ("cands": [set, logV, chain]):
  rules: verdict-only V(S); max over contained verified proofs (monotone closure); noisy-OR over contained proofs
  metrics: FSR of the arg-max set, ECE (10 bins), Brier score, AUROC -- raw, and after 2-fold cross-fitted
  isotonic recalibration (fit on half the questions, evaluate on the other half, then swap)
Also the retrieve-until-verified stopping risk: false-stop rate (stopped with an incomplete set) per threshold.

python calib.py musique_dev open_musique hotpotqa_dev open_hotpotqa 2wiki_dev open_2wiki > results/calib.json
"""
import json
import math
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from evaluate import gold_map, load

R = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")


def ece(p, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & (p < lo + 1 / bins if lo < 0.9 else p <= 1)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return 100 * e


def rule_scores(cands, k):
    pool = {}
    for j in range(1, k + 1):
        for T, v, *_ in cands.get(str(j), []):
            pool[frozenset(T)] = math.exp(v)
    out = []
    for S, v, ch in cands[str(k)]:
        S = frozenset(S)
        inside = [p for T, p in pool.items() if T <= S]
        out.append((S, ch, {"verdict": math.exp(v), "max": max(inside),
                            "noisy_or": 1 - float(np.prod([1 - min(p, 1 - 1e-6) for p in inside]))}))
    return out


def crossfit(p, y, seed=0):
    idx = np.random.default_rng(seed).permutation(len(p))
    a, b = idx[: len(p) // 2], idx[len(p) // 2:]
    q = np.zeros_like(p)
    for tr, te in ((a, b), (b, a)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr])
        q[te] = iso.predict(p[te])
    return q


def stats(p, y):
    return {"ECE": ece(p, y), "Brier": 100 * float(np.mean((p - y) ** 2)),
            "AUROC": 100 * roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan"),
            "mean_p": 100 * float(p.mean()), "FSR": 100 * float(y.mean())}


def main():
    res = {}
    for split in sys.argv[1:]:
        G = gold_map(split)
        rows = [r for r in load(f"{R}/{split}.{os.environ.get('CALIB_KEY', 'prover')}.jsonl") if r.get("cands")]
        res[split] = {}
        for k in (2, 3, 4):
            per = {"verdict": ([], []), "max": ([], []), "noisy_or": ([], [])}
            for r in rows:
                if str(k) not in r["cands"]:
                    continue
                g = set(G[r["id"]]["gold"])
                sc = rule_scores(r["cands"], k)
                for rule in per:
                    S, _, s = max(sc, key=lambda x: (x[2][rule], x[1]))
                    per[rule][0].append(s[rule]); per[rule][1].append(float(g <= S))
            res[split][k] = {}
            for rule, (p, y) in per.items():
                p, y = np.array(p), np.array(y)
                res[split][k][rule] = {"raw": stats(p, y), "isotonic": stats(crossfit(p, y), y)}
        # stopping risk with the deployed rule (noisy-OR), raw and recalibrated
        stop = []
        for tau in (0.5, 0.7, 0.9):
            fs_raw, fs_cal, n_raw, n_cal = [], [], [], []
            P = {}
            for k in range(1, 6):
                ps, ys, ids = [], [], []
                for r in rows:
                    if str(k) in r["cands"]:
                        g = set(G[r["id"]]["gold"])
                        S, _, s = max(rule_scores(r["cands"], k), key=lambda x: (x[2]["noisy_or"], x[1]))
                        ps.append(s["noisy_or"]); ys.append(float(g <= S)); ids.append(r["id"])
                ps, ys = np.array(ps), np.array(ys)
                cal = crossfit(ps, ys)
                for i, q in enumerate(ids):
                    P.setdefault(q, {})[k] = (ps[i], cal[i], ys[i])
            for q, d in P.items():
                for which, fs, nn in ((0, fs_raw, n_raw), (1, fs_cal, n_cal)):
                    kk = next((k for k in sorted(d) if d[k][which] >= tau), max(d))
                    nn.append(kk); fs.append(1 - d[kk][2])
            stop.append({"tau": tau, "false_stop_raw": 100 * np.mean(fs_raw), "avg_k_raw": float(np.mean(n_raw)),
                         "false_stop_cal": 100 * np.mean(fs_cal), "avg_k_cal": float(np.mean(n_cal))})
        res[split]["stop"] = stop
        print(split, "done", file=sys.stderr, flush=True)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
