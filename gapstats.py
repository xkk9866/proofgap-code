"""Analyses of the proof-gap field (CPU only).

  field   : quality of the predicted gap on every set the search read (rounded E[r] vs the true gap
            r = |G minus S|): accuracy, MAE, confusion, Spearman; and the monotonicity of the field along search
            edges (fraction of parent -> child edges where E[r] increases)
  descent : per depth, fraction of questions whose best frontier child truly reduces the gap; coverage kappa_k
  calib   : P(r=0) of the budget-k answer -- raw, and after an isotonic map fitted on the HELD-OUT training
            questions ({ds}_hold) -- ECE / Brier / AUROC; retrieve-until-complete stopping with the held-out
            calibrated probability >= 0.5 (avg #passages, FSR, false-stop rate)

python gapstats.py musique_dev gap musique_hold.gap
"""
import json
import math
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from evaluate import gold_map, load

R = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")


def field(split, key):
    G = gold_map(split)
    pr, tr, inc, n_edge = [], [], 0, 0
    for r in load(f"{R}/{split}.{key}.jsonl"):
        g = set(G[r["id"]]["gold"])
        if not g or not r.get("cands"):
            continue
        er = {}
        for k, cs in r["cands"].items():
            for S, lp, neg_er in cs:
                er[frozenset(S)] = -neg_er
                pr.append(-neg_er); tr.append(len(g - set(S)))
        for S, e in er.items():  # monotonicity along stored parent -> child edges
            for d in S:
                P = S - {d}
                if P in er:
                    n_edge += 1; inc += float(e > er[P] + 1e-6)
    pr, tr = np.array(pr), np.array(tr)
    rd = np.clip(np.rint(pr), 0, 4)
    conf = [[int(((tr == a) & (rd == b)).sum()) for b in range(5)] for a in range(5)]
    per = {int(a): {"n": int((tr == a).sum()), "acc": 100 * float((rd[tr == a] == a).mean()),
                    "mean_pred": float(pr[tr == a].mean())} for a in range(5) if (tr == a).any()}
    return {"acc": 100 * float((rd == tr).mean()), "mae": float(np.abs(pr - tr).mean()),
            "spearman": float(spearmanr(pr, tr).correlation), "confusion": conf, "per_gap": per,
            "monotone_violation": 100 * inc / max(1, n_edge), "edges": n_edge, "n_sets": int(len(tr))}


def descent(split, key):
    G = gold_map(split)
    out = {}
    for r in load(f"{R}/{split}.{key}.jsonl"):
        g = set(G[r["id"]]["gold"])
        if not g or not r.get("cands"):
            continue
        for k, cs in r["cands"].items():
            d = out.setdefault(int(k), {"cov": [], "best_gap": []})
            d["cov"].append(float(any(g <= set(x[0]) for x in cs)))
            best = min(cs, key=lambda x: -x[2])  # smallest E[r]
            d["best_gap"].append(len(g - set(best[0])))
    return {k: {"coverage": 100 * np.mean(v["cov"]), "true_gap_of_frontier_head": float(np.mean(v["best_gap"]))}
            for k, v in sorted(out.items())}


def _pk(split, key):
    G = gold_map(split)
    P = {}
    for r in load(f"{R}/{split}.{key}.jsonl"):
        g = set(G[r["id"]]["gold"])
        if not g:
            continue
        P[r["id"]] = {int(k): (math.exp(v), float(g <= set(r["sets"][k]))) for k, v in r["verdict"].items()}
    return P


def ece(p, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & (p < lo + 1 / bins if lo < 0.9 else p <= 1)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return 100 * e


def calib(split, key, hold):
    """hold: '<hold_split>.<key>' result of the same model on held-out training questions."""
    hs, hk = hold.split(".", 1)
    H = _pk(hs, hk)
    hp = np.array([v[0] for d in H.values() for v in d.values()])
    hy = np.array([v[1] for d in H.values() for v in d.values()])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(hp, hy)
    P = _pk(split, key)
    res = {}
    for k in (2, 3, 4):
        p = np.array([d[k][0] for d in P.values() if k in d]); y = np.array([d[k][1] for d in P.values() if k in d])
        q = iso.predict(p)
        res[k] = {w: {"ECE": ece(x, y), "Brier": 100 * float(np.mean((x - y) ** 2)),
                      "AUROC": 100 * roc_auc_score(y, x) if 0 < y.mean() < 1 else 50.0, "FSR": 100 * float(y.mean())}
                  for w, x in (("raw", p), ("heldout", q))}
    stop = []
    for tau in (0.5, 0.7, 0.9):
        for w in ("raw", "heldout"):
            ks, hit = [], []
            for d in P.values():
                kk = next((k for k in sorted(d) if (d[k][0] if w == "raw" else float(iso.predict([d[k][0]])[0])) >= tau), max(d))
                ks.append(kk); hit.append(d[kk][1])
            stop.append({"tau": tau, "prob": w, "avg_k": float(np.mean(ks)), "FSR": 100 * float(np.mean(hit)),
                         "false_stop": 100 * (1 - float(np.mean(hit)))})
    res["stop"] = stop
    return res


def mkadapt_heldout(split, key, hold, tau=0.5, kmax=5):
    """{split}.{key}_adapt.jsonl: smallest budget whose held-out-calibrated P(r=0) >= tau."""
    hs, hk = hold.split(".", 1)
    H = _pk(hs, hk)
    hp = np.array([v[0] for d in H.values() for v in d.values()])
    hy = np.array([v[1] for d in H.values() for v in d.values()])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(hp, hy)
    with open(f"{R}/{split}.{key}_adapt.jsonl", "w") as f:
        for r in load(f"{R}/{split}.{key}.jsonl"):
            ks = sorted(int(k) for k in r["sets"])
            kk = next((k for k in ks if float(iso.predict([math.exp(r["verdict"][str(k)])])[0]) >= tau), min(kmax, ks[-1]))
            f.write(json.dumps({"id": r["id"], "order": r["sets"][str(kk)], "k": kk}) + "\n")


if __name__ == "__main__":
    sp, key = sys.argv[1], sys.argv[2]
    out = {"field": field(sp, key), "descent": descent(sp, key)}
    if len(sys.argv) > 3:
        out["calib"] = calib(sp, key, sys.argv[3])
    print(json.dumps(out, indent=1, default=float))
