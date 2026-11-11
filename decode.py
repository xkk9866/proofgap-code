"""Offline test: decode the proof from every reading the search made.

python decode.py SPLIT PG.jsonl

Every candidate set S the search completed carries a consensus ladder (a distribution over its deficit r(S)) and one
necessity reading per member.  A hypothesis "the proof is T" fixes the label of every such reading:
    r(S) = |T minus S|,   necessity of t in S = 1[t in T].
score(T) = sum over the read sets S of  log P_S(r = |T minus S|) + sum_t log P_S(t necessary = 1[t in T]),
i.e. minus the training loss of the readings if T were the annotation.  Hypotheses are the visited candidates.
  dmap  length known (Set-EM): the best hypothesis of size k
  dsum  budget k: the size-k candidate covering the most posterior mass, sum_{T subset S} exp(score(T))
variants: l = ladders only, n = necessities only, b = both; suffix 1 = only the hypothesis' own readings (pi-like).
"""
import argparse
import math
from collections import defaultdict

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -500 else 0.0


def ls(x):
    return -math.log1p(math.exp(-x)) if x > -30 else x


def pmf(lad):
    cdf = [sig(u) for u in lad] + [1.0]
    return [math.log(max(cdf[0], 1e-6))] + [math.log(max(cdf[j] - cdf[j - 1], 1e-6)) for j in range(1, len(cdf))]


def scores(C, lad=True, nec=True, temp=1.0, depth_mean=False):
    allc = [x for k in C for x in C[k]]
    sets = [frozenset(x[0]) for x in allc]
    P = [pmf(x[3]) for x in allc]
    N = [{int(t): v for t, v in x[6].items()} if len(x) > 6 else {} for x in allc]
    out = []
    for T in sets:
        s = 0.0
        for S, p, n in zip(sets, P, N):
            w = 1.0 / len(C[str(len(S))]) if depth_mean else 1.0
            if lad:
                s += w * p[min(len(T - S), len(p) - 1)]
            if nec:
                s += w * sum(ls(v) if t in T else ls(-v) for t, v in n.items())
        out.append(s / temp)
    return allc, sets, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("pg")
    a = ap.parse_args()
    G = gold_map(a.split)
    var = {"l": dict(lad=True, nec=False), "n": dict(lad=False, nec=True), "b": dict(lad=True, nec=True),
           "bm": dict(lad=True, nec=True, depth_mean=True), "lm": dict(lad=True, nec=False, depth_mean=True)}
    fsr = defaultdict(list)
    hop = defaultdict(list)
    unk = defaultdict(list)
    n = 0
    for r in load(a.pg):
        e = G.get(r["id"])
        if not e or not e["gold"]:
            continue
        g = frozenset(e["gold"]); h = len(g); n += 1
        C = r["cands"]
        for name, kw in var.items():
            allc, sets, sc = scores(C, **kw)
            by = defaultdict(list)
            for T, s in zip(sets, sc):
                by[len(T)].append((s, T))
            dmap = {k: max(v)[1] for k, v in by.items()}
            mx = max(sc)
            dsum = {}
            for k, v in by.items():
                best, bv = None, -1.0
                for _, S in v:
                    m = sum(math.exp(s - mx) for T, s in zip(sets, sc) if T <= S)
                    if m > bv:
                        best, bv = S, m
                dsum[k] = best
            for rule, d in (("dmap", dmap), ("dsum", dsum)):
                for k in range(2, 6):
                    S = d.get(k, d[max(d)])
                    fsr[(name, rule, k)].append(float(g <= S))
                S = d.get(h, frozenset())
                fsr[(name, rule, "EM")].append(float(g <= S))
                hop[(name, rule, h)].append(float(g <= S))
            top = sets[max(range(len(sc)), key=lambda i: sc[i])]
            unk[name].append((float(g <= top), float(g == top), len(top)))
    print(f"{a.split} n={n}")
    for name in var:
        for rule in ("dmap", "dsum"):
            print(f"  {name:2s} {rule} FSR@2-5 " + "/".join(f"{100 * sum(fsr[(name, rule, k)]) / n:.1f}" for k in range(2, 6))
                  + f"  Set-EM {100 * sum(fsr[(name, rule, 'EM')]) / n:.1f}  by hops "
                  + " ".join(f"{hh}:{100 * sum(v) / len(v):.1f}" for (nn, rr, hh), v in sorted(hop.items()) if nn == name and rr == rule))
        u = unk[name]
        print(f"  {name:2s} unknown length: complete {100 * sum(x[0] for x in u) / n:.1f}  exact {100 * sum(x[1] for x in u) / n:.1f}"
              f"  passages {sum(x[2] for x in u) / n:.2f}")


if __name__ == "__main__":
    main()
