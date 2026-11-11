"""Proof-disjunctive decision for PROVER.

A set S is complete iff it contains at least one complete proof.  Every lattice node T is a candidate
proof whose completeness the verdict estimates (V(T) = P(T is complete | q, T): the AND over requirements,
checked jointly by the encoder).  Treating the verified candidate proofs contained in S as independent
witnesses gives the noisy-OR completeness of S:

    P_or(S) = 1 - prod_{T in L, T subset of S} (1 - V(T))

For every budget k PROVER returns argmax_{S in C_k} P_or(S) (ties: proposal score).  P_or is monotone in S,
reduces to V(S) when S contains no smaller verified proof, and rewards a set that hedges an uncertain
requirement by containing two alternative proofs.

python decide.py musique_dev prover     # re-decides an output file in place; verdict-only copy -> prover_vonly
"""
import json
import math
import os
import sys


def or_decide(cands):
    """cands: {k(str): [[set, logV, proposal], ...]} -> sets, logP_or, proposal per k."""
    sets, ver, prop = {}, {}, {}
    for k in sorted(cands, key=int):
        pool = {}
        for j in range(1, int(k) + 1):
            for x in cands.get(str(j), []):  # x = [set, log V, proposal, ...]
                pool[frozenset(x[0])] = x[1]

        def log1m_por(S):
            S = set(S)
            return sum(math.log1p(-min(math.exp(v), 1 - 1e-6)) for T, v in pool.items() if T <= S)

        best = min(cands[k], key=lambda x: (log1m_por(x[0]), -x[2]))
        l1m = log1m_por(best[0])
        sets[k], prop[k] = sorted(best[0]), best[2]
        ver[k] = math.log(max(1e-12, -math.expm1(l1m)))
    return sets, ver, prop


def log_pi(x):
    """log P(T is exactly the annotated proof) of a candidate x = [set, log P(r = 0), chain, ..., {member: necessity logit}]:
    no passage is missing (consensus rung 0) and every member is necessary (one reading per view)."""
    return x[1] + sum(-math.log1p(math.exp(-v)) if v > -30 else v for v in x[6].values())


def pi_decide(cands):
    """Decision from the posterior over which candidate is the proof.

    The proof is one set, so the events {G = T} are disjoint and P(G subset of S) = sum_{T subset of S} P(G = T).
    Budget k: the depth-k candidate that covers the most posterior mass (ties: proposal score).
    Returns sets, log covered mass, proposal per k, and the single most probable proof with its log pi (the answer
    when the length of the proof is not given)."""
    sets, ver, prop = {}, {}, {}
    allc = [x for k in cands for x in cands[k]]
    mass = [(frozenset(x[0]), math.exp(log_pi(x))) for x in allc]
    for k in sorted(cands, key=int):
        sub = [(T, p) for T, p in mass if len(T) <= int(k)]

        def covered(S):
            S = set(S)
            return sum(p for T, p in sub if T <= S)

        best = max(cands[k], key=lambda x: (covered(x[0]), x[2]))
        sets[k], prop[k] = sorted(best[0]), best[2]
        ver[k] = math.log(min(1.0, max(1e-12, covered(best[0]))))
    top = max(allc, key=lambda x: (log_pi(x), x[2]))
    return sets, ver, prop, sorted(top[0]), round(log_pi(top), 4)


def redecide(split, key, R="/data/yaokang/ercrag/results"):
    p = f"{R}/{split}.{key}.jsonl"
    rows = [json.loads(l) for l in open(p)]
    vonly = f"{R}/{split}.{key}_vonly.jsonl"
    if not os.path.exists(vonly):
        with open(vonly, "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "cands"}) + "\n")
    with open(p, "w") as f:
        for r in rows:
            if r.get("cands"):
                r["sets"], r["verdict"], r["pred"] = or_decide(r["cands"])
                ks = sorted(r["sets"], key=int)
                r["order"] = r["sets"][ks[-1]]
                r["decision"] = "or"
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    redecide(sys.argv[1], sys.argv[2])
