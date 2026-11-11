"""Every number of the paper from the stored selections (no GPU).  Writes $RES/collect36.json and prints a summary.

python collect36.py
"""
import json
import math
import os

import numpy as np

from boot import per_question
from evaluate import gold_map, load, qa_metrics, retrieval_metrics

R = os.environ.get("RES", "/data/yaokang/ercrag/results_pg4")
METHODS = ["dense", "rerank", "qwen3rr", "rankgpt", "ircot", "llmsetfill", "beam", "PG"]
SPLITS = ["musique_dev", "2wiki_dev", "hotpotqa_dev", "open_musique", "open_2wiki", "open_hotpotqa", "open100_musique", "open100_2wiki", "musique_hold"]
EXTRA = {  # other selections of interest: split -> keys
    "musique_dev": ["musique_nl_bin_s1", "musique_nl_holo_s1", "musique_nolist2_s0", "musique_nolist2_s1", "musique_nolist2_s2",
                    "musique_v6", "pg_ep1", "pg_holo", "pg_bin", "pg_noopen", "pg_nolist_w4", "pg_nolist_fl", "pg_nolist_w1", "pg_nolist_e2", "pg_nolist_e8",
                    "musique_nolist_s1", "musique_nolist_s2", "musique_nolist_a2", "musique_nl_bin", "musique_nl_holo", "musique_nl_noopen", "musique_nolist_large",
                    "pg_long_ep1", "pg_long_ep4", "beam_w3", "beam_w4"],
    "open_musique": ["musique_nl_bin_s1", "musique_nl_holo_s1", "musique_nolist2_s0", "musique_nolist2_s1", "musique_nolist2_s2", "musique_v6", "pg_ep1", "pg_holo", "pg_bin", "pg_noopen", "musique_nolist_s1", "musique_nolist_s2", "musique_nolist_a2", "musique_nl_bin",
                     "musique_nl_holo", "musique_nl_noopen", "musique_nolist_large"],
    "musique_hold": ["musique_nolist_s1", "musique_nolist_s2", "musique_nolist_a2", "pg"],
    "hotpotqa_dev": ["pg_nolist_xmu", "beam_xmu"],
    "2wiki_dev": ["pg_nolist_xmu", "beam_xmu"],
}


def pgkey(split):
    return "2wiki_nolist" if split in ("2wiki_dev", "open_2wiki") else "pg_nolist"


def path(split, key):
    return f"{R}/{split}.{pgkey(split) if key == 'PG' else key}.jsonl"


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -500 else 0.0


def stop_stats(split, p, tau=0.5):
    G = gold_map(split)
    rows = [r for r in load(p) if G[r["id"]]["gold"]]
    if not rows or "ladder" not in rows[0]:
        return None
    n = len(rows)
    comp = exact = length = enc = encfull = unsat = 0
    for r in rows:
        g = set(G[r["id"]]["gold"])
        S, kk, ok = [], 0, False
        for k in range(1, 6):
            if str(k) not in r["ladder"]:
                continue
            S, kk = r["sets"][str(k)], k
            if sig(r["ladder"][str(k)][0]) >= tau:
                ok = True
                break
        comp += g <= set(S); exact += g == set(S); length += len(S); unsat += not ok
        enc += r["cost"].get(str(kk), 0)
        encfull += r["cost"].get("4", r["cost"].get(str(kk), 0))
    return {"complete": 100 * comp / n, "exact": 100 * exact / n, "passages": length / n, "enc": enc / n, "enc_to4": encfull / n,
            "unsat": 100 * unsat / n, "n": n}


def boot(split, pa, pb, n=10000):
    A, B = per_question(split, pa), per_question(split, pb)
    ids = sorted(set(A) & set(B))
    d = np.array([A[i] for i in ids]) - np.array([B[i] for i in ids])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(n, len(ids)))
    out = {}
    for j, nm in enumerate(["FSR@2", "FSR@3", "FSR@4", "FSR@5", "SetEM"]):
        bs = d[idx, j].mean(1) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[nm] = {"diff": 100 * d[:, j].mean(), "lo": lo, "hi": hi, "p": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                   "wins": int((d[:, j] > 0).sum()), "losses": int((d[:, j] < 0).sum())}
    return out


def main():
    out = {"ret": {}, "stop": {}, "qa": {}, "boot": {}, "seeds": {}, "mtime": {}}
    cj = f"{R}/collect36.json"
    cache = json.load(open(cj)) if os.path.exists(cj) else {}
    for sp in SPLITS:
        for key in METHODS + EXTRA.get(sp, []):
            p = path(sp, key)
            if not os.path.exists(p):
                continue
            ck = f"{sp}|{key}"
            if ck in cache.get("ret", {}) and cache.get("mtime", {}).get(ck) == os.path.getmtime(p):
                out["ret"][ck] = cache["ret"][ck]
                out["mtime"][ck] = cache["mtime"][ck]
                if ck in cache.get("stop", {}):
                    out["stop"][ck] = cache["stop"][ck]
                continue
            try:
                m = retrieval_metrics(sp, p, ks=(2, 3, 4, 5), by_hops=True)
            except Exception as e:  # a file still being written
                print("skip", p, e)
                continue
            if "FSR@2" not in m or m.get("n", 0) < 900:  # empty or partial file
                print("skip (partial)", p, m.get("n"))
                continue
            out["ret"][ck] = m
            out["mtime"][ck] = os.path.getmtime(p)
            print(f"{sp:16s} {key:20s} FSR " + "/".join(f"{m[f'FSR@{k}']:.1f}" for k in (2, 3, 4, 5)) + f"  SE {m['SetEM@|G|']:.1f}  SR@4 {m['SR@4']:.1f}  n {m['n']}"
                  + "  hops " + " ".join(f"{h}:{v:.1f}" for h, v in m["by_hops"].items() if h.split("hop@")[0] == h.split("@")[1]), flush=True)
            if key == "PG" or key.startswith("musique_no") or key.startswith("musique_nl") or key.startswith("pg_nolist"):
                s = stop_stats(sp, p)
                if s:
                    out["stop"][f"{sp}|{key}"] = s
                    print(f"    stop: complete {s['complete']:.1f} exact {s['exact']:.1f} passages {s['passages']:.2f} enc {s['enc']:.0f} (to 4: {s['enc_to4']:.0f}) unsat {s['unsat']:.1f}")
        json.dump(out, open(cj + ".part", "w"), indent=1)
        pa, pb = path(sp, "PG"), path(sp, "beam")
        if sp in cache.get("boot", {}):
            out["boot"][sp] = cache["boot"][sp]
        elif os.path.exists(pa) and os.path.exists(pb):
            out["boot"][sp] = boot(sp, pa, pb)
            b = out["boot"][sp]["SetEM"]
            print(f"    boot PG - Beam Set-EM {b['diff']:+.1f} [{b['lo']:+.1f}, {b['hi']:+.1f}] p={b['p']:.4f}")
    # seeds: Beam s1 / s2 live in sub-directories
    for s in (1, 2):
        for sp in ("musique_dev", "open_musique"):
            p = f"{R}/s{s}/{sp}.beam.jsonl"
            if os.path.exists(p):
                m = retrieval_metrics(sp, p, ks=(2, 3, 4, 5), by_hops=True)
                out["seeds"][f"{sp}|beam|s{s}"] = m
                print(f"seed {s} {sp} beam FSR " + "/".join(f"{m[f'FSR@{k}']:.1f}" for k in (2, 3, 4, 5)) + f" SE {m['SetEM@|G|']:.1f}")
    for f in sorted(os.listdir(R)):
        if f.startswith("qa_") and f.endswith("_q.jsonl"):
            sp, key = f[3:-6].split(".", 1)
            try:
                out["qa"][f"{sp}|{key}"] = qa_metrics(sp, f"{R}/{f}")
            except Exception as e:
                print("skip qa", f, e)
                continue
            q = out["qa"][f"{sp}|{key}"]
            print(f"qa {sp:14s} {key:22s} EM {q['EM']:.1f} F1 {q['F1']:.1f} n {q['n']}")
    json.dump(out, open(f"{R}/collect36.json", "w"), indent=1)
    print("COLLECT_DONE")


if __name__ == "__main__":
    main()
