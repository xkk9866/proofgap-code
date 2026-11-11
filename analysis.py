"""Analyses for PROVER outputs (no GPU needed).

  decomp    : coverage kappa_k / verification error eps_k of the propose-then-verify decision
              (Proposition 1), for the verdict decision and for the proposal-score (chain) decision
  adaptive  : retrieve-until-verified curve: smallest k whose verdict >= tau  -> (avg #passages, FSR)
  boot      : paired bootstrap p-value of FSR@k / Set-EM differences between two result files

python analysis.py decomp musique_dev prover
python analysis.py adaptive musique_dev prover
python analysis.py boot musique_dev prover beam_w2
"""
import json
import sys

import numpy as np

from evaluate import gold_map, load

R = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")
D = "/data/yaokang/ercrag/data/proc"


def _sel(r, k):
    if "sets" in r:
        return r["sets"].get(str(k), r["order"][:k])
    return r["order"][:k]


def decomp(split, key, ks=(2, 3, 4, 5)):
    G = gold_map(split)
    rows = {r["id"]: r for r in load(f"{R}/{split}.{key}.jsonl")}
    out = {}
    for k in ks:
        cov, ver, chain, orr = [], [], [], []
        for qid, e in G.items():
            r = rows.get(qid)
            if r is None or not e["gold"] or str(k) not in r.get("cands", {}):
                continue
            g = set(e["gold"])
            c = r["cands"][str(k)]
            cov.append(float(any(g <= set(x[0]) for x in c)))
            ver.append(float(g <= set(max(c, key=lambda x: (x[1], x[2]))[0])))
            chain.append(float(g <= set(max(c, key=lambda x: x[2])[0])))
            orr.append(float(g <= set(r["sets"][str(k)])))
        cov, ver, chain, orr = map(np.array, (cov, ver, chain, orr))
        out[k] = {"coverage": 100 * cov.mean(), "FSR_verdict": 100 * ver.mean(), "FSR_chain": 100 * chain.mean(),
                  "eps_verdict": 100 * (cov - ver).mean(), "eps_chain": 100 * (cov - chain).mean(),
                  "FSR_or": 100 * orr.mean(), "eps_or": 100 * (cov - orr).mean(),
                  "cand_per_q": float(np.mean([len(r["cands"].get(str(k), [])) for r in rows.values()])), "n": len(cov)}
    return out


def adaptive(split, key, kmax=5):
    G = gold_map(split)
    rows = {r["id"]: r for r in load(f"{R}/{split}.{key}.jsonl")}
    curve = []
    for tau in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01]:
        n, hit = [], []
        for qid, e in G.items():
            r = rows.get(qid)
            if r is None or not e["gold"]:
                continue
            kk = kmax
            for k in range(1, kmax + 1):
                if str(k) in r["verdict"] and np.exp(r["verdict"][str(k)]) >= tau:
                    kk = k
                    break
            n.append(kk); hit.append(float(set(e["gold"]) <= set(_sel(r, kk))))
        curve.append({"tau": tau, "avg_k": float(np.mean(n)), "FSR": 100 * float(np.mean(hit))})
    return curve


def boot(split, a, b, B=2000, seed=0):
    G = gold_map(split)
    ra = {r["id"]: r for r in load(f"{R}/{split}.{a}.jsonl")}
    rb = {r["id"]: r for r in load(f"{R}/{split}.{b}.jsonl")}
    ids = [q for q, e in G.items() if q in ra and q in rb and e["gold"]]
    rng = np.random.default_rng(seed)
    res = {}
    for k in ["2", "3", "4", "5", "G"]:
        def hit(r, e):
            kk = len(e["gold"]) if k == "G" else int(k)
            return float(set(e["gold"]) <= set(_sel(r, kk)))
        d = np.array([hit(ra[q], G[q]) - hit(rb[q], G[q]) for q in ids])
        bs = rng.integers(0, len(d), (B, len(d)))
        means = d[bs].mean(1)
        res[k] = {"diff": 100 * d.mean(), "p": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}
    return res


def mkadapt(split, key, tau=0.5, kmax=5):
    """Write {split}.{key}_adapt.jsonl: the smallest-budget set whose verdict reaches tau (else budget kmax)."""
    with open(f"{R}/{split}.{key}_adapt.jsonl", "w") as f:
        for r in load(f"{R}/{split}.{key}.jsonl"):
            kk = next((k for k in range(1, kmax + 1) if str(k) in r["verdict"] and np.exp(r["verdict"][str(k)]) >= tau), kmax)
            kk = min(kk, max(int(k) for k in r["sets"]))
            f.write(json.dumps({"id": r["id"], "order": _sel(r, kk), "k": kk}) + "\n")


def mkadapt_cal(split, key, tau=0.5, kmax=5, seed=0):
    """{split}.{key}_cal.jsonl: smallest budget whose CALIBRATED completeness probability reaches tau.

    The aggregate score P is mapped to a probability by isotonic regression cross-fitted over questions
    (fit on one half of the questions, all budgets; apply to the other half), so the threshold 1/2 is
    applied to a probability estimated on questions disjoint from the ones being decided.
    """
    from sklearn.isotonic import IsotonicRegression
    G = {json.loads(l)["id"]: set(json.loads(l)["gold"]) for l in open(f"{D}/{split}.jsonl")}
    rows = list(load(f"{R}/{split}.{key}.jsonl"))
    fold = np.random.default_rng(seed).permutation(len(rows)) % 2
    pk = [{int(k): (float(np.exp(v)), float(G[r["id"]] <= set(r["sets"][k]))) for k, v in r["verdict"].items()} for r in rows]
    iso = []
    for f_ in (0, 1):
        tr = [v for i, d in enumerate(pk) if fold[i] != f_ for v in d.values()]
        iso.append(IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit([a for a, _ in tr], [b for _, b in tr]))
    with open(f"{R}/{split}.{key}_cal.jsonl", "w") as f:
        for i, r in enumerate(rows):
            ks = sorted(k for k in pk[i] if k <= kmax)
            cal = iso[fold[i]].predict([pk[i][k][0] for k in ks])
            kk = next((k for k, c in zip(ks, cal) if c >= tau), ks[-1])
            f.write(json.dumps({"id": r["id"], "order": _sel(r, kk), "k": kk, "p": float(cal[ks.index(kk)])}) + chr(10))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "decomp":
        print(json.dumps(decomp(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "adaptive":
        print(json.dumps(adaptive(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "mkadapt":
        mkadapt(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    elif cmd == "mkadapt_cal":
        mkadapt_cal(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    elif cmd == "boot":
        print(json.dumps(boot(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
