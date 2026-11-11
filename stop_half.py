"""Unknown proof length: return the shortest decided set whose consensus P(r = 0) is at least 1/2, else the depth-5 set.

python stop_half.py SPLIT PG.jsonl [--beam BEAM.jsonl] [--write OUT.jsonl]
Prints completeness, mean passages and mean encoder sequences of the stopped search; --write stores {id, order}.
"""
import argparse
import json
import math
from collections import defaultdict

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x))


def stopped(r, kmax=5):
    chosen = None
    for k in range(1, kmax + 1):
        if str(k) not in r.get("ladder", {}):
            continue
        chosen = (k, r["sets"][str(k)], (r.get("cost") or {}).get(str(k), 0) or 0)
        if sig(r["ladder"][str(k)][0]) >= 0.5:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("pg")
    ap.add_argument("--beam", default="")
    ap.add_argument("--write", default="")
    a = ap.parse_args()
    G = gold_map(a.split)
    P = {r["id"]: r for r in load(a.pg)}
    rows, out = [], []
    full = []
    for qid, e in G.items():
        if not e["gold"] or qid not in P:
            continue
        g = set(e["gold"]); h = e.get("hops") or len(g)
        k, S, c = stopped(P[qid])
        rows.append((float(g <= set(S)), k, c, h))
        full.append((P[qid].get("cost") or {}).get("4", 0) or 0)
        out.append({"id": qid, "order": list(S)})
    n = len(rows)
    hops = defaultdict(list)
    for y, _, _, h in rows:
        hops[h].append(y)
    print(f"{a.split} stop@0.5 complete {100 * sum(r[0] for r in rows) / n:.1f} passages {sum(r[1] for r in rows) / n:.2f} "
          f"enc {sum(r[2] for r in rows) / n:.0f} (to depth 4: {sum(full) / n:.0f}) hops "
          + " ".join(f"{h}:{100 * sum(v) / len(v):.1f}" for h, v in sorted(hops.items())))
    if a.beam:
        B = {r["id"]: r for r in load(a.beam)}
        for k in (2, 3, 4):
            hit = []
            for qid, e in G.items():
                if not e["gold"] or qid not in P:
                    continue
                b = B.get(qid, {})
                sets = b.get("sets") or {}
                S = set(sets[str(k)]) if str(k) in sets else set((b.get("order") or [])[:k])
                hit.append(float(set(e["gold"]) <= S))
            print(f"  beam fixed {k}: {100 * sum(hit) / len(hit):.1f}")
    if a.write:
        with open(a.write, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print("wrote", a.write)


if __name__ == "__main__":
    main()
