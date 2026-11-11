"""Necessity readings of a ProofGap v6 search output (cands carry {member: necessity logit} as 7th element).

python ana_nec.py SPLIT PG.jsonl [--write RULE OUT.jsonl]

pi(T) = P(r(T) = 0) * prod_t P(t necessary | T) is the probability that T is exactly the annotated proof (complete
and irredundant).  The proof is unique, so the events {T is the proof} are disjoint and
    P(G subset of S) = sum_{T in C, T subset of S} pi(T).
Decisions compared at every budget k over the depth-k candidates C_k:
  or     stored noisy-OR of P(r = 0) over contained candidates (decide.or_decide)
  p0     P(r(S) = 0) of the set alone
  mp     pi(S) of the set alone
  pis    sum of pi over contained candidates (P(S contains the proof))
Unknown length: argmax_T pi(T) over every candidate of every size (no threshold), against the shortest decided set
with P(r = 0) >= 1/2.  Also: AUC of the necessity logit (proof members vs others) and how often the intruder of a
near-proof (|T & G| = |G| - 1 at k = |G|) is its least necessary member.
"""
import argparse
import json
import math
from collections import defaultdict

from evaluate import gold_map, load

RULES = ("or", "p0", "mp", "pis", "ed", "er", "es")


def ls(x):
    return -math.log1p(math.exp(-x)) if x > -30 else x


def sig(x):
    return 1 / (1 + math.exp(-x))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(1 for t in range(i, j) if allv[t][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def logpi(x):
    return x[1] + sum(ls(v) for v in x[6].values())


def decide(r, rule):
    """{k: set} for every depth k under one rule."""
    C = r["cands"]
    out = {}
    for k in sorted(C, key=int):
        L = C[k]
        if rule == "or":
            out[k] = r["sets"][k]
        elif rule == "p0":
            out[k] = max(L, key=lambda x: (x[1], x[2]))[0]
        elif rule == "mp":
            out[k] = max(L, key=lambda x: (logpi(x), x[2]))[0]
        elif rule == "ed":  # smallest expected edit distance E[r] + E[s]
            out[k] = min(L, key=lambda x: (sum(sig(-u) for u in x[3]) + sum(sig(-v) for v in x[6].values()), -x[2]))[0]
        elif rule == "er":  # smallest expected deficit
            out[k] = min(L, key=lambda x: (sum(sig(-u) for u in x[3]), -x[2]))[0]
        elif rule == "es":  # smallest expected surplus
            out[k] = min(L, key=lambda x: (sum(sig(-v) for v in x[6].values()), -x[2]))[0]
        else:
            sub = [(frozenset(x[0]), math.exp(logpi(x))) for j in C if int(j) <= int(k) for x in C[j]]
            out[k] = max(L, key=lambda x: (sum(p for T, p in sub if T <= set(x[0])), x[2]))[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("pg")
    ap.add_argument("--write", nargs=2, default=None, metavar=("RULE", "OUT"))
    a = ap.parse_args()
    G = gold_map(a.split)
    P = {r["id"]: r for r in load(a.pg)}
    fsr = defaultdict(list)
    hop_em = defaultdict(list)
    pos, neg, loc = [], [], []
    mapr, halfr, rows_out = [], [], []
    for qid, e in G.items():
        if not e["gold"] or qid not in P:
            continue
        r, g = P[qid], set(e["gold"])
        h = len(g)
        C = r["cands"]
        for L in C.values():
            for x in L:
                for t, v in x[6].items():
                    (pos if int(t) in g else neg).append(v)
        dec = {rule: decide(r, rule) for rule in RULES}
        for rule, d in dec.items():
            for k in range(2, 6):
                S = d.get(str(k), d[max(d, key=int)] if d else [])
                fsr[(rule, k)].append(float(g <= set(S)))
            S = d.get(str(h), [])
            fsr[(rule, "EM")].append(float(g <= set(S)))
            hop_em[(rule, h)].append(float(g <= set(S)))
        fsr[("covered", "EM")].append(float(any(set(x[0]) == g for x in C.get(str(h), []))))
        for x in C.get(str(h), []):
            T = set(x[0])
            if len(T & g) == h - 1 and len(T) == h:
                intr = (T - g).pop()
                low = min(x[6], key=lambda t: x[6][t])
                loc.append(float(int(low) == intr))
        allc = [x for L in C.values() for x in L]
        best = max(allc, key=lambda x: (logpi(x), x[2]))
        mapr.append((float(g <= set(best[0])), float(set(best[0]) == g), len(best[0]), r["cost"].get(str(len(best[0])), 0)))
        chosen, ck = None, 0
        for k in range(1, 6):
            if str(k) not in r.get("ladder", {}):
                continue
            chosen, ck = r["sets"][str(k)], k
            if sig(r["ladder"][str(k)][0]) >= 0.5:
                break
        halfr.append((float(g <= set(chosen)), float(set(chosen) == g), len(chosen), r["cost"].get(str(ck), 0)))
        if a.write:
            d = dec[a.write[0]]
            ks = sorted(d, key=int)
            rows_out.append({"id": qid, "sets": d, "order": d[ks[-1]], "map": best[0]})
    n = len(fsr[("or", "EM")])
    print(f"{a.split} n={n} covered@|G| {100 * sum(fsr[('covered', 'EM')]) / n:.1f}")
    for rule in RULES:
        print(f"  {rule:4s} FSR@2-5 " + "/".join(f"{100 * sum(fsr[(rule, k)]) / n:.1f}" for k in range(2, 6))
              + f"  Set-EM {100 * sum(fsr[(rule, 'EM')]) / n:.1f}  by hops "
              + " ".join(f"{h}:{100 * sum(v) / len(v):.1f}" for (rr, h), v in sorted(hop_em.items()) if rr == rule))
    print(f"  necessity AUC (proof members vs others) {auc(pos, neg):.3f}  n+={len(pos)} n-={len(neg)}")
    if loc:
        print(f"  near-proofs at k=|G|: intruder is the least necessary member {100 * sum(loc) / len(loc):.1f}% (n={len(loc)})")
    for name, rows in (("argmax pi", mapr), ("stop@1/2", halfr)):
        m = len(rows)
        print(f"  unknown length {name}: complete {100 * sum(x[0] for x in rows) / m:.1f}  exact {100 * sum(x[1] for x in rows) / m:.1f}"
              f"  passages {sum(x[2] for x in rows) / m:.2f}  enc {sum(x[3] for x in rows) / m:.0f}")
    if a.write:
        with open(a.write[1], "w") as f:
            for x in rows_out:
                f.write(json.dumps(x) + "\n")
        print("wrote", a.write[1])


if __name__ == "__main__":
    main()
