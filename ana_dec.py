"""Decision and stopping analyses on a stored search (no GPU).

python ana_dec.py SPLIT PG.jsonl

(1) Which depth-k candidate is returned: its own completeness (p0), the noisy-OR over the read sets it contains
    (or), or the monotone envelope max over the read sets it contains (max).
(2) Stopping: return the shortest decided set whose completeness reaches tau; sweep of tau.
(3) Deficit as a progress signal: mean true deficit of the best candidate at each depth, per hop count.
"""
import math
import sys

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -500 else 0.0


def decide(cands, k, rule):
    pool = {}
    for j in range(1, k + 1):
        for x in cands.get(str(j), []):
            pool[frozenset(x[0])] = x[1]

    def score(x):
        S = set(x[0])
        if rule == "p0":
            return x[1]
        sub = [v for T, v in pool.items() if T <= S]
        if rule == "max":
            return max(sub)
        return -sum(math.log1p(-min(math.exp(v), 1 - 1e-6)) for v in sub)  # or

    best = max(cands[str(k)], key=lambda x: (score(x), x[2]))
    return sorted(best[0]), score(best)


def main():
    split, path = sys.argv[1:3]
    G = gold_map(split)
    rows = [r for r in load(path) if G[r["id"]]["gold"]]
    n = len(rows)
    print(f"{split}: {n} questions")
    for rule in ("p0", "or", "max"):
        em, fs, hop = 0, {k: 0 for k in range(2, 6)}, {}
        for r in rows:
            g = set(G[r["id"]]["gold"])
            h = len(g)
            for k in range(2, 6):
                if str(k) in r["cands"]:
                    S, _ = decide(r["cands"], k, rule)
                else:
                    S = r["order"]
                ok = g <= set(S)
                fs[k] += ok
                if k == h:
                    em += ok
                    hop.setdefault(h, []).append(ok)
        print(f"  decide {rule:3s} FSR@2-5 " + "/".join(f"{100 * fs[k] / n:.1f}" for k in range(2, 6)) + f"  Set-EM {100 * em / n:.1f}  hops "
              + "/".join(f"{100 * sum(v) / len(v):.1f}" for _, v in sorted(hop.items())))
    print("  stop at the shortest returned set with completeness >= tau:")
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        comp = exact = length = enc = unsat = 0
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
            comp += g <= set(S); exact += g == set(S); length += len(S); enc += r["cost"].get(str(kk), 0); unsat += not ok
        print(f"    tau {tau:.1f}: complete {100 * comp / n:.1f}  exact {100 * exact / n:.1f}  passages {length / n:.2f}  enc {enc / n:.0f}  never satisfied {100 * unsat / n:.1f}")
    # oracle length and fixed lengths for reference
    for k in (2, 3, 4, 5):
        comp = sum(set(G[r["id"]]["gold"]) <= set(r["sets"].get(str(k), r["order"])) for r in rows)
        print(f"    fixed k={k}: complete {100 * comp / n:.1f}  passages {k}")
    comp = sum(set(G[r["id"]]["gold"]) <= set(r["sets"].get(str(len(G[r['id']]['gold'])), r["order"])) for r in rows)
    print(f"    oracle length |G|: complete {100 * comp / n:.1f}  passages {sum(len(G[r['id']]['gold']) for r in rows) / n:.2f}")
    # expected deficit of the returned set vs its true deficit, by depth
    print("  returned set at depth k: predicted E[r] vs true deficit")
    for k in range(1, 6):
        pe, te, m = 0.0, 0.0, 0
        for r in rows:
            if str(k) not in r["ladder"]:
                continue
            g = set(G[r["id"]]["gold"])
            lad = r["ladder"][str(k)]
            pe += sum(sig(-u) for u in lad)
            te += len(g - set(r["sets"][str(k)]))
            m += 1
        if m:
            print(f"    k={k}: E[r] {pe / m:.2f}  true {te / m:.2f}  (n={m})")


if __name__ == "__main__":
    main()
