"""Paired bootstrap of answer EM / F1: the reader given ProofGap's passages against the reader given each other
retriever's passages (same questions, same reader).  No GPU.

python qaboot.py            -> $RES/qaboot.json
"""
import glob
import json
import os

import numpy as np

from evaluate import f1, gold_map, load, norm

R = os.environ.get("RES", "/data/yaokang/ercrag/results_pg4")
REF = os.environ.get("QA_REF", "pg_nolist_q")


def per_q(split, p):
    G = gold_map(split)
    out = {}
    for r in load(p):
        ans = G[r["id"]]["answers"]
        out[r["id"]] = (max(float(norm(r["pred"]) == norm(a)) for a in ans), max(f1(r["pred"], a) for a in ans))
    return out


def main():
    res = {}
    rng = np.random.default_rng(0)
    for split in ("musique_dev", "open_musique"):
        ref = f"{R}/qa_{split}.{REF}.jsonl"
        if not os.path.exists(ref):
            print("no reference", ref)
            continue
        A = per_q(split, ref)
        for p in sorted(glob.glob(f"{R}/qa_{split}.*_q.jsonl")):
            key = os.path.basename(p)[len(f"qa_{split}."):-len(".jsonl")]
            if key == REF:
                continue
            B = per_q(split, p)
            ids = sorted(set(A) & set(B))
            d = np.array([A[i] for i in ids]) - np.array([B[i] for i in ids])  # (n, 2): EM, F1
            idx = rng.integers(0, len(ids), size=(10000, len(ids)))
            o = {"n": len(ids)}
            for j, nm in enumerate(("EM", "F1")):
                bs = d[idx, j].mean(1) * 100
                lo, hi = np.percentile(bs, [2.5, 97.5])
                o[nm] = {"diff": 100 * d[:, j].mean(), "lo": lo, "hi": hi, "p": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))}
            res[f"{split}|{key}"] = o
            print(f"{split:14s} {REF} - {key:22s} n {len(ids)}  EM {o['EM']['diff']:+.1f} [{o['EM']['lo']:+.1f}, {o['EM']['hi']:+.1f}] p={o['EM']['p']:.3f}"
                  f"   F1 {o['F1']['diff']:+.1f} [{o['F1']['lo']:+.1f}, {o['F1']['hi']:+.1f}] p={o['F1']['p']:.3f}", flush=True)
    json.dump(res, open(f"{R}/qaboot.json", "w"), indent=1)
    print("QABOOT_DONE")


if __name__ == "__main__":
    main()
