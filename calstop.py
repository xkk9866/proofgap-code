"""Calibration and adaptive stopping on the sets the retriever actually returned (CPU only).

For every split and budget k, the score is P(hat S_k) stored in r["verdict"][k] and the label is whether
hat S_k = r["sets"][k] is complete, so FSR here equals the main-table FSR exactly.
  - raw ECE / Brier / AUROC of the score
  - cross-fitted isotonic map (fit on one half of the QUESTIONS, all budgets pooled; apply to the other half)
  - retrieve-until-complete: smallest k with (raw | calibrated) probability >= tau

python calstop.py ladder musique_dev hotpotqa_dev open_musique open_hotpotqa open_2wiki  -> results/calstop_ladder.json
"""
import json
import math
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from evaluate import gold_map, load

R = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")
TAUS = (0.3, 0.5, 0.7, 0.9)


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return 100 * sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any())


def stats(p, y):
    return {"ECE": ece(p, y), "Brier": 100 * float(np.mean((p - y) ** 2)),
            "AUROC": 100 * roc_auc_score(y, p) if 0 < y.mean() < 1 else 50.0, "FSR": 100 * float(y.mean())}


def run(split, key, kmax=5, seed=0):
    G = gold_map(split)
    rows = []
    for r in load(f"{R}/{split}.{key}.jsonl"):
        g = set(G[r["id"]]["gold"])
        rows.append({int(k): (math.exp(v), float(bool(g) and g <= set(r["sets"][k]))) for k, v in r["verdict"].items() if int(k) <= kmax})
    fold = np.random.default_rng(seed).permutation(len(rows)) % 2
    cal = [dict() for _ in rows]
    for f_ in (0, 1):
        tr = [v for i, d in enumerate(rows) if fold[i] != f_ for v in d.values()]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit([a for a, _ in tr], [b for _, b in tr])
        for i, d in enumerate(rows):
            if fold[i] == f_:
                ks = sorted(d)
                cal[i] = dict(zip(ks, iso.predict([d[k][0] for k in ks])))
    out = {}
    for k in (2, 3, 4):
        m = [i for i, d in enumerate(rows) if k in d]
        p = np.array([rows[i][k][0] for i in m]); y = np.array([rows[i][k][1] for i in m])
        q = np.array([cal[i][k] for i in m])
        out[str(k)] = {"raw": stats(p, y), "cal": stats(q, y)}
    stop = []
    for w in ("raw", "cal"):
        for tau in TAUS:
            ks, hit = [], []
            for d, c in zip(rows, cal):
                sc = {k: (d[k][0] if w == "raw" else c[k]) for k in d}
                kk = next((k for k in sorted(d) if sc[k] >= tau), max(d))
                ks.append(kk); hit.append(d[kk][1])
            stop.append({"prob": w, "tau": tau, "avg_k": float(np.mean(ks)), "FSR": 100 * float(np.mean(hit))})
    out["stop"] = stop
    return out


if __name__ == "__main__":
    key = sys.argv[1]
    res = {sp: run(sp, key) for sp in sys.argv[2:]}
    json.dump(res, open(f"{R}/calstop_{key}.json", "w"), indent=1)
    print(json.dumps({sp: v["stop"] for sp, v in res.items()}, indent=0)[:3000])
