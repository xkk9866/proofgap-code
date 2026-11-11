"""Paired bootstrap over questions: difference in FSR@k / Set-EM between two result files on the same split.

python boot.py SPLIT A.jsonl B.jsonl [--n 10000]
"""
import argparse
import json

import numpy as np

from evaluate import fsr, gold_map, load


def per_question(split, path, ks=(2, 3, 4, 5)):
    G = gold_map(split)
    rows = {r["id"]: r for r in load(path)}
    out = {}
    for qid, e in G.items():
        if not e["gold"]:
            continue
        g = e["gold"]
        r = rows.get(qid)
        if r is None:
            out[qid] = [0.0] * (len(ks) + 1)
            continue
        o, sets = r["order"], r.get("sets")
        pick = (lambda k: sets.get(str(k), o)) if sets else (lambda k: o[:k])
        out[qid] = [fsr(pick(k), g, k) for k in ks] + [fsr(pick(len(g)), g, len(g))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    A, B = per_question(a.split, a.a), per_question(a.split, a.b)
    ids = sorted(set(A) & set(B))
    xa = np.array([A[i] for i in ids]); xb = np.array([B[i] for i in ids])
    d = xa - xb
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(a.n, len(ids)))
    names = ["FSR@2", "FSR@3", "FSR@4", "FSR@5", "SetEM"]
    res = {"split": a.split, "n": len(ids)}
    for j, nm in enumerate(names):
        bs = d[idx, j].mean(1) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        res[nm] = {"a": 100 * xa[:, j].mean(), "b": 100 * xb[:, j].mean(), "diff": 100 * d[:, j].mean(),
                   "lo": lo, "hi": hi, "p": p, "wins": int((d[:, j] > 0).sum()), "losses": int((d[:, j] < 0).sum())}
        print(f"{a.split:14s} {nm:6s} {res[nm]['a']:.1f} vs {res[nm]['b']:.1f}  diff {res[nm]['diff']:+.1f} "
              f"[{lo:+.1f}, {hi:+.1f}] p={p:.4f}  wins {res[nm]['wins']} losses {res[nm]['losses']}")
    print("JSON", json.dumps(res))


if __name__ == "__main__":
    main()
