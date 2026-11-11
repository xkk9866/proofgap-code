"""Calibration of the two readings of a ProofGap search output.

python cal.py SPLIT PG.jsonl [--bins 10] [--json OUT.json]

Every candidate set T the search read is one sample.
  completeness   sigma(u_0(T))                       against 1[G subset of T]
  exact proof    pi(T) = P(r = 0) prod_t P(t in G)   against 1[T = G]
  deficit        E[r(T)] = sum_h sigma(-u_h)         against |G minus T|
  surplus        E[s(T)] = sum_t sigma(-nec_t)       against |T minus G|
  edit distance  E[r] + E[s]                         against |T symmetric-difference G|
Reports expected calibration error (equal-width bins), the reliability table, mean absolute error and the rank
correlation of the predicted distance, and the total posterior mass sum_T pi(T) a question assigns to its candidates
(the events {G = T} are disjoint, so a calibrated reader keeps this at or below one).
"""
import argparse
import json
import math

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -500 else 0.0


def ls(x):
    return -math.log1p(math.exp(-x)) if x > -30 else x


def ece(pairs, bins):
    tab = [[0, 0.0, 0.0] for _ in range(bins)]
    for p, y in pairs:
        b = min(bins - 1, int(p * bins))
        tab[b][0] += 1
        tab[b][1] += p
        tab[b][2] += y
    n = len(pairs)
    e = sum(abs(t[1] - t[2]) for t in tab) / max(1, n)
    rows = [(i / bins, (i + 1) / bins, t[0], t[1] / t[0] if t[0] else 0.0, t[2] / t[0] if t[0] else 0.0) for i, t in enumerate(tab)]
    return e, rows


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j < len(o) and v[o[j]] == v[o[i]]:
                j += 1
            for t in range(i, j):
                r[o[t]] = (i + j - 1) / 2
            i = j
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("pg")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    G = gold_map(a.split)
    comp, exact, dec = [], [], []
    er, es, ed, tr, ts, td = [], [], [], [], [], []
    mass = []
    for r in load(a.pg):
        e = G.get(r["id"])
        if not e or not e["gold"]:
            continue
        g = set(e["gold"])
        tot = 0.0
        for k, L in r["cands"].items():
            for x in L:
                T = set(x[0])
                p0 = sig(x[3][0])
                nec = list(x[6].values()) if len(x) > 6 else []
                pi = math.exp(x[1] + sum(ls(v) for v in nec))
                comp.append((p0, float(g <= T)))
                exact.append((pi, float(T == g)))
                tot += pi
                r_hat = sum(sig(-u) for u in x[3])
                s_hat = sum(sig(-v) for v in nec)
                er.append(r_hat); es.append(s_hat); ed.append(r_hat + s_hat)
                tr.append(len(g - T)); ts.append(len(T - g)); td.append(len(g ^ T))
        mass.append(tot)
        h = str(len(g))
        if h in r["sets"]:
            S = set(r["sets"][h])
            x = next((c for c in r["cands"][h] if set(c[0]) == S), None)
            if x is not None:
                dec.append((sig(x[3][0]), float(g <= S)))
    out = {"split": a.split, "n_sets": len(comp), "n_q": len(mass)}
    print(f"{a.split}: {len(comp)} candidate sets of {len(mass)} questions")
    for name, pairs in (("completeness", comp), ("exact proof", exact), ("returned set at |G|", dec)):
        e, rows = ece(pairs, a.bins)
        base = sum(y for _, y in pairs) / len(pairs)
        brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
        print(f"  {name:20s} ECE {100 * e:.2f}  Brier {brier:.4f}  base rate {100 * base:.1f}  n={len(pairs)}")
        print("     bins " + " ".join(f"[{lo:.1f},{hi:.1f}) n={n} p={p:.2f} y={y:.2f};" for lo, hi, n, p, y in rows if n))
        out[name] = {"ece": e, "brier": brier, "base": base, "rows": rows}
    for name, ph, tt in (("deficit", er, tr), ("surplus", es, ts), ("edit distance", ed, td)):
        mae = sum(abs(p - t) for p, t in zip(ph, tt)) / len(tt)
        rho = spearman(ph, tt)
        print(f"  {name:20s} MAE {mae:.3f}  Spearman {rho:.3f}  mean true {sum(tt) / len(tt):.2f}")
        out[name] = {"mae": mae, "spearman": rho}
    m = sum(mass) / len(mass)
    over = sum(1 for v in mass if v > 1.0) / len(mass)
    print(f"  posterior mass over candidates: mean {m:.3f}  above one for {100 * over:.1f}% of questions")
    out["mass"] = {"mean": m, "over": over}
    if a.json:
        json.dump(out, open(a.json, "w"))


if __name__ == "__main__":
    main()
