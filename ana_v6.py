"""Offline diagnosis of a ProofGap search output (focus_search format).

python ana_v6.py SPLIT RESULT.jsonl
  failure anatomy at k = |G| (coverage vs decision, deficit of the returned set, is the intruder the weakest view)
  decision rules on the same candidates: noisy-OR (stored), own P0, monotone envelope, envelope + parent bound
"""
import json
import math
import sys
from collections import Counter, defaultdict

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x))


def main(split, path):
    G = gold_map(split)
    rows = {r["id"]: r for r in load(path)}
    stat = Counter()
    weak = Counter()
    rules = defaultdict(lambda: defaultdict(list))
    for qid, e in G.items():
        if not e["gold"] or qid not in rows:
            continue
        g = set(e["gold"]); n = len(g)
        r = rows[qid]
        C = {k: [(frozenset(c[0]), c) for c in L] for k, L in r["cands"].items()}
        ladder = {}
        for k, L in C.items():
            for T, c in L:
                ladder[T] = [sig(u) for u in c[3]]  # F_T(h) = P(r <= h), h = 0..R-2

        def env(T):
            return max(ladder[S][0] for S in ladder if S <= T)

        def parent_cap(T):
            caps = [ladder[T - {t}][1] for t in T if (T - {t}) in ladder and len(ladder[T - {t}]) > 1]
            return min(caps) if caps else 1.0

        for k in range(2, 6):
            if str(k) not in C:
                continue
            cand = C[str(k)]
            stored = set(r["sets"].get(str(k), []))
            v0 = max(cand, key=lambda x: (x[1][1], x[1][2]))[0]
            en = max(cand, key=lambda x: (env(x[0]), x[1][1]))[0]
            ec = max(cand, key=lambda x: (min(env(x[0]), max(ladder[x[0]][0], parent_cap(x[0]))), x[1][1]))[0]
            for name, S in (("noisyOR", stored), ("ownP0", v0), ("envelope", en), ("env+cap", ec)):
                rules[name][k].append(float(g <= set(S)))
            if k == n:
                for name, S in (("noisyOR", stored), ("ownP0", v0), ("envelope", en), ("env+cap", ec)):
                    rules[name]["SE"].append(float(g <= set(S)))
        k = str(n)
        if k not in C:
            stat["no depth"] += 1
            continue
        S = set(r["sets"][k])
        covered = any(T == frozenset(g) for T, _ in C[k])
        if g <= S:
            stat["hit"] += 1
            continue
        d = len(g - S)
        stat[f"miss covered={covered} deficit={d}"] += 1
        if d == 1:
            c = next(c for T, c in C[k] if T == frozenset(S))
            views = {int(t): v for t, v in c[5].items()}
            intr = [t for t in S if t not in g]
            if intr and views:
                wmin = min(views, key=views.get)
                weak["intruder is weakest view" if wmin in intr else "a gold member is weakest"] += 1
                par = frozenset(S - set(intr))
                inlat = par in ladder
                weak[f"correct parent read={inlat}"] += 1
    tot = sum(v for kk, v in stat.items() if kk != "no depth")
    print(f"== {split}  n={tot}")
    for kk, v in sorted(stat.items(), key=lambda x: -x[1]):
        print(f"  {kk:40s} {v:5d}  {100 * v / tot:5.1f}%")
    for kk, v in weak.items():
        print(f"  near-proof misses: {kk:30s} {v}")
    for name, d in rules.items():
        cols = " ".join(f"FSR@{k} {100 * sum(d[k]) / len(d[k]):.1f}" for k in range(2, 6) if d[k])
        print(f"  {name:9s} Set-EM {100 * sum(d['SE']) / len(d['SE']):.1f}  {cols}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
