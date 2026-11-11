"""Stopping without the proof length: the deficit of ProofGap against the sequence score of Beam Retrieval.

python ana_stopcmp.py SPLIT out.json  name=kind:path ...        kind is "pg" or "beam"

ProofGap returns the shortest decided set whose completeness P(r = 0) reaches tau (the paper uses tau = 1/2).
Beam Retrieval has no stopping rule of its own, so it is given the two rules its score allows:
  cont  keep extending while the best sequence of the next depth has probability >= t (t = 1/2 is the rule without
        a tuned constant: "the continuation is more likely valid than not"); at least one passage is returned.
  peak  return the depth (>= 2) whose best sequence has the highest probability.
For every rule: % of questions whose annotated support is inside the returned set, % whose returned set is exactly
the support, and the mean size.  Thresholds are swept so that the two models can be compared at the same mean size.
"""
import json
import math
import os
import sys

from evaluate import gold_map, load


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -500 else 0.0


def summarise(rows):
    n = len(rows)
    return {"complete": 100 * sum(r[0] for r in rows) / n, "exact": 100 * sum(r[1] for r in rows) / n,
            "size": sum(r[2] for r in rows) / n, "f1": 100 * sum(r[3] for r in rows) / n, "n": n}


def score(g, S):
    """(support inside the returned set, returned set is the support, size, F1 of the returned set against the support)"""
    tp = len(g & S)
    return (float(g <= S), float(g == S), len(S), 2 * tp / (len(g) + len(S)) if S else 0.0)


def pg_rule(r, tau):
    chosen = []
    for k in range(1, 6):
        if str(k) not in r.get("ladder", {}):
            continue
        chosen = r["sets"][str(k)]
        if sig(r["ladder"][str(k)][0]) >= tau:
            break
    return chosen


def beam_cont(r, t):
    ks = sorted(int(k) for k in r["sets"])
    chosen = r["sets"][str(ks[0])]
    for k in ks[1:]:
        if math.exp(r["pred"][str(k)]) < t:
            break
        chosen = r["sets"][str(k)]
    return chosen


def beam_peak(r):
    ks = [k for k in sorted(int(k) for k in r["sets"]) if k >= 2]
    if not ks:
        return r["order"]
    return r["sets"][str(max(ks, key=lambda k: r["pred"][str(k)]))]


def beam_cliff(r, kmin=1):
    """the depth after which the score of the best sequence falls the most"""
    ks = sorted(int(k) for k in r["sets"])
    cand = [k for k in ks if k >= kmin and k + 1 in ks]
    if not cand:
        return r["order"]
    return r["sets"][str(max(cand, key=lambda k: r["pred"][str(k)] - r["pred"][str(k + 1)]))]


def beam_cum(r, t):
    """keep extending while the product of the step probabilities stays >= t"""
    ks = sorted(int(k) for k in r["sets"])
    chosen, acc = r["sets"][str(ks[0])], r["pred"][str(ks[0])]
    for k in ks[1:]:
        acc += r["pred"][str(k)]
        if math.exp(acc) < t:
            break
        chosen = r["sets"][str(k)]
    return chosen


def interp(curve, size):
    """completeness of a (size, complete) curve at a given mean size, linear between the two nearest points"""
    pts = sorted(curve)
    if size <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= size <= x1:
            return y0 + (y1 - y0) * (size - x0) / max(x1 - x0, 1e-9)
    return pts[-1][1]


def main():
    split, outp = sys.argv[1:3]
    G = gold_map(split)
    out = {}
    grid = [i / 20 for i in range(1, 20)]
    for arg in sys.argv[3:]:
        name, rest = arg.split("=", 1)
        kind, path = rest.split(":", 1)
        R = {r["id"]: r for r in load(path)}
        ids = [i for i in G if i in R and G[i]["gold"]]

        def ev(rule):
            rows = []
            for i in ids:
                g, S = set(G[i]["gold"]), set(rule(R[i]))
                rows.append(score(g, S))
            return summarise(rows)
        def ev_hops(rule):  # the same, split by the length of the proof
            o_ = {}
            for h in (2, 3, 4):
                rows = []
                for i in ids:
                    if len(G[i]["gold"]) == h:
                        g, S = set(G[i]["gold"]), set(rule(R[i]))
                        rows.append(score(g, S))
                if rows:
                    o_[h] = summarise(rows)
            return o_
        o = {"kind": kind, "fixed": {k: ev(lambda r, k=k: r["sets"].get(str(k), r["order"])) for k in (2, 3, 4, 5)}}
        o["oracle_length"] = ev(lambda r: r["sets"].get(str(len(G[r["id"]]["gold"])), r["order"]))
        main_rules = {"rule": (lambda r: pg_rule(r, 0.5))} if kind == "pg" else {"peak": beam_peak, "cliff": beam_cliff, "cliff2": (lambda r: beam_cliff(r, 2))}
        o["by_hops"] = {nm: ev_hops(fn) for nm, fn in main_rules.items()}
        for nm, bh in o["by_hops"].items():
            print(f"   {name} {nm} by hops (complete, exact, size): " + "  ".join(f"{h}-hop {v['complete']:.1f}, {v['exact']:.1f}, {v['size']:.2f}" for h, v in bh.items()))
        if kind == "pg":
            o["sweep"] = {f"{t:.2f}": ev(lambda r, t=t: pg_rule(r, t)) for t in grid}
            o["rule"] = o["sweep"]["0.50"]
        else:
            r0 = R[ids[0]]
            o["depths"] = sorted(int(k) for k in r0["sets"])
            o["sweep"] = {f"{t:.2f}": ev(lambda r, t=t: beam_cont(r, t)) for t in grid}
            o["rule"] = o["sweep"]["0.50"]
            o["peak"] = ev(beam_peak)
            o["cliff"] = ev(beam_cliff)
            o["cliff2"] = ev(lambda r: beam_cliff(r, 2))
            o["cum"] = {f"{t:.2f}": ev(lambda r, t=t: beam_cum(r, t)) for t in grid}
            best = max(list(o["sweep"].values()) + list(o["cum"].values()) + [o["peak"], o["cliff"], o["cliff2"]], key=lambda v: v["exact"])
            o["best_exact_any_rule"] = best
            print(f"   {name}: cliff complete {o['cliff']['complete']:.1f} exact {o['cliff']['exact']:.1f} size {o['cliff']['size']:.2f} | cliff from depth 2: complete {o['cliff2']['complete']:.1f} exact {o['cliff2']['exact']:.1f} size {o['cliff2']['size']:.2f}"
                  f" | best exact over every rule and threshold: {best['exact']:.1f} (complete {best['complete']:.1f}, size {best['size']:.2f})")
            print("   cumulative-probability rule (threshold: size, complete, exact): " + "  ".join(f"{t}: {v['size']:.2f}, {v['complete']:.1f}, {v['exact']:.1f}" for t, v in o["cum"].items()))
            # how often the score of the next depth says "stop" exactly at the end of the proof
            ps = [math.exp(R[i]["pred"][str(len(G[i]["gold"]) + 1)]) for i in ids if str(len(G[i]["gold"]) + 1) in R[i]["pred"]]
            pe = [math.exp(R[i]["pred"][str(len(G[i]["gold"]))]) for i in ids if str(len(G[i]["gold"])) in R[i]["pred"]]
            o["mean_p_at_proof_length"] = sum(pe) / max(len(pe), 1)
            o["mean_p_one_past_proof"] = sum(ps) / max(len(ps), 1)
        out[name] = o
        # one line per rule: complete / exact / F1 / size.  "best" rows pick the threshold with hindsight on these questions.
        fams = {"rule at 1/2": o["rule"], "oracle length": o["oracle_length"]}
        for k in (2, 3, 4):
            fams[f"fixed {k}"] = o["fixed"][k]
        # the length predicted from the question alone (lenclf.py), classifier seeds 0-2: LENCLF=prefix of {prefix}.{split}.s{seed}.json
        pre = os.environ.get("LENCLF")
        if pre:
            for cs in (0, 1, 2):
                p = f"{pre}.{split}.s{cs}.json"
                if os.path.exists(p):
                    L = json.load(open(p))
                    o[f"lenclf{cs}"] = ev(lambda r, L=L: r["sets"].get(str(L[r["id"]]), r["order"]))
                    o[f"lenclf{cs}_acc"] = 100 * sum(L[i] == len(G[i]["gold"]) for i in ids) / len(ids)
                    o[f"lenclf{cs}_by_hops"] = ev_hops(lambda r, L=L: r["sets"].get(str(L[r["id"]]), r["order"]))
                    fams[f"question-only length, clf seed {cs}"] = o[f"lenclf{cs}"]
                    print(f"   {name}: length classifier seed {cs} accuracy {o[f'lenclf{cs}_acc']:.1f}; by hops (complete, exact, size): "
                          + "  ".join(f"{h}-hop {v['complete']:.1f}, {v['exact']:.1f}, {v['size']:.2f}" for h, v in o[f"lenclf{cs}_by_hops"].items()))
        for crit in ("f1", "exact"):
            fams[f"threshold swept, best {crit}"] = max(o["sweep"].values(), key=lambda v: v[crit])
        if kind == "beam":
            for nm in ("peak", "cliff", "cliff2"):
                fams[nm] = o[nm]
            for crit in ("f1", "exact"):
                fams[f"cumulative swept, best {crit}"] = max(o["cum"].values(), key=lambda v: v[crit])
            pool_ = [v for k, v in fams.items() if k != "oracle length" and not k.startswith("question-only")]  # rules of its own score
            o["best_f1_any_rule"] = max(pool_, key=lambda v: v["f1"])
            o["best_exact_any_rule"] = max(pool_, key=lambda v: v["exact"])
        for nm, v in fams.items():
            print(f"TAB {split} {name} | {nm:32s} | complete {v['complete']:5.1f} | exact {v['exact']:5.1f} | F1 {v['f1']:5.1f} | size {v['size']:.2f}")
        print(f"{name} ({kind}, n={o['rule']['n']}): rule at 1/2: complete {o['rule']['complete']:.1f} exact {o['rule']['exact']:.1f} size {o['rule']['size']:.2f}"
              + (f" | peak: complete {o['peak']['complete']:.1f} exact {o['peak']['exact']:.1f} size {o['peak']['size']:.2f}"
                 f" | p at proof length {o['mean_p_at_proof_length']:.2f}, one past {o['mean_p_one_past_proof']:.2f}, depths {o['depths']}" if kind == "beam" else ""))
        print("   fixed budgets: " + "  ".join(f"@{k} {o['fixed'][k]['complete']:.1f}" for k in (2, 3, 4, 5)))
        print("   sweep (threshold: size, complete, exact): " + "  ".join(f"{t}: {v['size']:.2f}, {v['complete']:.1f}, {v['exact']:.1f}" for t, v in o["sweep"].items()))
    # every ProofGap run at its rule against every Beam run at the same mean size (its swept curve and its fixed budgets)
    for a, oa in out.items():
        if oa["kind"] != "pg":
            continue
        for b, ob in out.items():
            if ob["kind"] != "beam":
                continue
            sz = oa["rule"]["size"]
            sw = max(interp([(v["size"], v["complete"]) for v in ob[c].values()], sz) for c in ("sweep", "cum"))
            fx = interp([(k, v["complete"]) for k, v in ob["fixed"].items()], sz)
            print(f"{a} rule {oa['rule']['complete']:.1f} at {sz:.2f} passages | {b} at that size: swept score rule {sw:.1f}, fixed budget {fx:.1f}")
    json.dump(out, open(outp, "w"))


if __name__ == "__main__":
    main()
