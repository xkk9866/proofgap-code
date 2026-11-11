"""Where does a ProofGap search lose the proof?  python ana_err.py SPLIT PG.jsonl

For every question: is the annotated proof among the depth-|G| candidates (covered), where does it rank under the
completeness rung, how far is the returned set from it, and what would the best of the top-m candidates give.
"""
import math
import sys
from collections import Counter, defaultdict

from evaluate import gold_map, load


def main():
    split, path = sys.argv[1:3]
    G = gold_map(split)
    n = 0
    cov = Counter(); em = Counter(); rank = Counter(); dist = Counter(); hopn = Counter()
    topm = defaultdict(int)
    marg = []
    union2 = 0
    first_lost = Counter()
    for r in load(path):
        e = G.get(r["id"])
        if not e or not e["gold"]:
            continue
        g = frozenset(e["gold"]); h = len(g); n += 1; hopn[h] += 1
        C = r["cands"]
        L = C.get(str(h), [])
        S = frozenset(r["sets"].get(str(h), []))
        ok = S == g
        em[h] += ok
        order = sorted(L, key=lambda x: (x[1], x[2]), reverse=True)
        sets = [frozenset(x[0]) for x in order]
        covered = g in sets
        cov[h] += covered
        for m in (1, 2, 3, 4, 8):
            topm[m] += g in sets[:m]
        if covered:
            rk = sets.index(g)
            rank[min(rk, 5)] += 1
            if not ok:
                dist[len(S - g)] += 1
                x = next(c for c in L if frozenset(c[0]) == S)
                marg.append(x[1] - order[rk][1])
                if len(sets) >= 2 and g <= (sets[0] | sets[1]):
                    union2 += 1
        else:
            # first depth at which no candidate is a subset of the proof
            lost = None
            for k in range(1, h + 1):
                if not any(frozenset(c[0]) <= g for c in C.get(str(k), [])):
                    lost = k
                    break
            first_lost[lost] += 1
    tot = sum(em.values())
    print(f"{split} n={n} Set-EM {100 * tot / n:.1f}  covered {100 * sum(cov.values()) / n:.1f}")
    for h in sorted(hopn):
        print(f"  {h}-hop n={hopn[h]}  Set-EM {100 * em[h] / hopn[h]:.1f}  covered {100 * cov[h] / hopn[h]:.1f}")
    print("  proof within the top-m candidates by completeness: " + "  ".join(f"m={m}: {100 * v / n:.1f}" for m, v in sorted(topm.items())))
    c = sum(rank.values())
    print("  rank of the proof when covered: " + "  ".join(f"{k + 1}{'+' if k == 5 else ''}: {100 * v / c:.1f}%" for k, v in sorted(rank.items())))
    d = sum(dist.values())
    if d:
        print(f"  covered but another set returned: {100 * d / n:.1f}% of questions; intruders in the returned set: "
              + "  ".join(f"{k}: {100 * v / d:.1f}%" for k, v in sorted(dist.items())))
        marg.sort()
        print(f"  log P0 margin of the returned set over the proof: median {marg[len(marg) // 2]:.2f}  "
              f"below 0.5 for {100 * sum(1 for m in marg if m < 0.5) / d:.1f}%  below 0.1 for {100 * sum(1 for m in marg if m < 0.1) / d:.1f}%")
        print(f"  proof inside the union of the two best candidates: {100 * union2 / d:.1f}% of those")
    u = sum(first_lost.values())
    if u:
        print(f"  not covered: {100 * u / n:.1f}% of questions; depth at which the last subset of the proof left the candidates: "
              + "  ".join(f"{k}: {100 * v / u:.1f}%" for k, v in sorted(first_lost.items(), key=lambda kv: (kv[0] is None, kv[0]))))


if __name__ == "__main__":
    main()
