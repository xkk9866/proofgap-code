"""Does the deficit know when the pool is not enough?  Deficit-driven pool growth.

python ana_pool.py SPLIT_A PG_A.jsonl SPLIT_B PG_B.jsonl

A is the small first-stage pool (top 30), B the larger pool of the same questions (top 100).  The search on A ends
with a completeness reading for the set it returns.  If no depth reaches P(r = 0) >= 1/2 the deficit says passages
are still missing, and the question is searched again in B.  Reports, for A alone, B alone and the deficit-driven
mixture: Set-EM, FSR@k, the stop-rule completeness and length, the mean pool size and the encoder sequences; and how
well the completeness reached on A predicts that A really contains the proof (AUROC).
"""
import math
import sys

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -500 else 0.0


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


def stop(r):
    """(set, depth, satisfied) under the rule: shortest decided set with P(r = 0) >= 1/2, else the deepest."""
    chosen, ck, ok = [], 0, False
    for k in range(1, 6):
        if str(k) not in r.get("ladder", {}):
            continue
        chosen, ck = r["sets"][str(k)], k
        if sig(r["ladder"][str(k)][0]) >= 0.5:
            ok = True
            break
    return chosen, ck, ok


def main():
    sa, fa, sb, fb = sys.argv[1:5]
    GA, GB = gold_map(sa), gold_map(sb)
    A = {r["id"]: r for r in load(fa)}
    B = {r["id"]: r for r in load(fb)}
    ids = [i for i in GA if i in A and i in B and i in GB and GA[i]["gold"]]
    rows = {"A": [], "B": [], "mix": []}
    conf_in, conf_out, conf_ok, conf_bad = [], [], [], []
    moved = 0
    for i in ids:
        ga, gb = set(GA[i]["gold"]), set(GB[i]["gold"])
        na, nb = len(GA[i]["ctxs"]), len(GB[i]["ctxs"])
        in_a = all(g < na for g in ga)
        ra, rb = A[i], B[i]
        Sa, ka, oka = stop(ra)
        Sb, kb, okb = stop(rb)
        best = max(sig(v[0]) for v in ra["ladder"].values())
        (conf_in if in_a else conf_out).append(best)
        (conf_ok if ga <= set(Sa) else conf_bad).append(best)
        h = len(ga)

        def pack(r, g, S, k, pool, extra_cost=0):
            return dict(em=float(g <= set(r["sets"].get(str(h), []))),
                        fsr={kk: float(g <= set(r["sets"].get(str(kk), r["order"]))) for kk in range(2, 6)},
                        comp=float(g <= set(S)), length=len(S), pool=pool,
                        enc=r["cost"].get(str(k), 0) + extra_cost)
        pa, pb = pack(ra, ga, Sa, ka, na), pack(rb, gb, Sb, kb, nb)
        rows["A"].append(pa); rows["B"].append(pb)
        if oka:
            rows["mix"].append(pa)
        else:
            moved += 1
            rows["mix"].append(pack(rb, gb, Sb, kb, nb, extra_cost=ra["cost"].get(str(ka), 0)))
    n = len(ids)
    print(f"{sa} -> {sb}: n={n}; proof in A {100 * len(conf_in) / n:.1f}%; searched again in B: {100 * moved / n:.1f}%")
    for name in ("A", "B", "mix"):
        L = rows[name]
        print(f"  {name:3s} Set-EM {100 * sum(x['em'] for x in L) / n:.1f}  FSR@2-5 "
              + "/".join(f"{100 * sum(x['fsr'][k] for x in L) / n:.1f}" for k in range(2, 6))
              + f"  stop: complete {100 * sum(x['comp'] for x in L) / n:.1f} in {sum(x['length'] for x in L) / n:.2f} passages"
              + f"  pool {sum(x['pool'] for x in L) / n:.1f}  enc {sum(x['enc'] for x in L) / n:.0f}")
    print(f"  AUROC of the best completeness reached on A: proof is in A {auc(conf_in, conf_out):.3f}; returned set is complete {auc(conf_ok, conf_bad):.3f}")
    # the same question asked of other scores on pool A: expected deficit of the deepest set, and (optional 5th argument)
    # the sequence score of a chain retriever at each depth
    er_in, er_out = [], []
    for i in ids:
        lad = A[i]["ladder"]
        k = max(lad, key=int)
        (er_in if all(g < len(GA[i]["ctxs"]) for g in GA[i]["gold"]) else er_out).append(-sum(sig(-u) for u in lad[k]))
    print(f"  AUROC of minus the expected deficit of the deepest set: proof is in A {auc(er_in, er_out):.3f}")
    if len(sys.argv) > 5:
        Bm = {r["id"]: r for r in load(sys.argv[5])}
        for k in ("2", "3", "4", "max", "mean"):
            pin, pout = [], []
            for i in ids:
                if i not in Bm:
                    continue
                pr = Bm[i]["pred"]
                v = pr.get(k) if k in pr else (max(pr[x] for x in ("2", "3", "4") if x in pr) if k == "max" else sum(pr[x] for x in ("2", "3", "4") if x in pr) / 3)
                (pin if all(g < len(GA[i]["ctxs"]) for g in GA[i]["gold"]) else pout).append(v)
            print(f"  chain retriever, sequence score at depth {k}: proof is in A {auc(pin, pout):.3f}")


if __name__ == "__main__":
    main()
