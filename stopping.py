"""Budget as a decision: cost--quality of stopping rules on the outputs of ProofGap's proof search (CPU only).

The search computes depth k only after depth k-1, and every output row stores, for each depth k, the decided set
S_k, its pooled completeness score P(S_k) ("verdict"), its holistic deficit ladder (u_0, u_1, u_2) ("ladder") and
the cumulative number of encoder sequences after depth k ("cost").  Stopping at depth k therefore costs exactly
cost[k] encoder sequences and |S_k| passages for the reader, so every rule below is evaluated without re-running.

Calibration uses only the SOURCE hold-out questions (training questions excluded by train.py --holdout; no labels of
the evaluated split), fitted by isotonic regression:
    c0 : P(S_k)            -> Pr[S_k complete]
    ch : sigma(u_h(S_k))   -> Pr[S_{k+h} complete]           (h = 1, 2; the ladder's h-step reachability)
Rules (utility = 1[complete] - lam * passages):
    fixed k           return S_k
    threshold tau     first k with c0 >= tau                    (needs only a completeness verdict)
    lookahead lam     first k with  max_h [ max(c_h, c0) - c0 - lam * h ] <= 0
                      i.e. no horizon h whose expected gain exceeds the price of h more passages; this is the
                      h-step look-ahead Bayes rule, and only a graded deficit supplies c_1 and c_2.
    oracle            first k whose set is complete (upper bound)
Also reports reader tokens (Qwen3 tokenizer) of the returned sets.

python stopping.py --res results_pg/musique_dev.pg.jsonl --split musique_dev \
                   --hold results_pg/musique_hold.pg.jsonl --hold_split musique_hold --out results_pg/stop_musique_dev.json
"""
import argparse
import json
import math

import numpy as np
from sklearn.isotonic import IsotonicRegression

D = "/data/yaokang/ercrag/data/proc"
QTOK = "/home/yaokang/models/Qwen3-8B-AWQ"
K = 5


def load(p):
    return [json.loads(l) for l in open(p)]


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def features(res_path, split):
    G = {e["id"]: e for e in load(f"{D}/{split}.jsonl")}
    out = []
    for r in load(res_path):
        e = G.get(r["id"])
        if e is None or not e["gold"] or "ladder" not in r:
            continue
        g = set(e["gold"])
        ks = sorted(int(k) for k in r["sets"] if int(k) <= K)
        row = {"id": r["id"], "ks": ks}
        for k in ks:
            u = r["ladder"][str(k)]  # informative cumulative logits u_0 .. u_{R-2} of the decided set
            row[k] = {"p": math.exp(r["verdict"][str(k)]), "r": {h: sig(u[h]) for h in range(1, len(u))},
                      "y": float(g <= set(r["sets"][str(k)])), "set": r["sets"][str(k)], "cost": r.get("cost", {}).get(str(k), 0)}
        out.append(row)
    return out


def fit(rows):
    """Isotonic calibrators fitted on hold-out rows: c[0] for completeness now, c[h] for completeness h steps later."""
    xs, ys = {0: []}, {0: []}
    for r in rows:
        for k in r["ks"]:
            xs[0].append(r[k]["p"]); ys[0].append(r[k]["y"])
            for h, v in r[k]["r"].items():
                if k + h in r:
                    xs.setdefault(h, []).append(v); ys.setdefault(h, []).append(r[k + h]["y"])
    return {h: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xs[h], ys[h]) for h in xs if len(xs[h]) > 1}


def xfit(rows, seed=0):
    """Target cross-fitting (2 folds over questions): an upper reference that uses target labels."""
    fold = np.random.default_rng(seed).permutation(len(rows)) % 2
    cals = [fit([r for i, r in enumerate(rows) if fold[i] != f]) for f in (0, 1)]
    return [cals[fold[i]] for i in range(len(rows))]


LAMS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)


def stop_depth(r, c, rule, param):
    """Depth at which `rule` stops for one question (calibrators c)."""
    kk = r["ks"][-1]
    for k in r["ks"]:
        if rule == "fixed":
            if k == param:
                return k
            continue
        if rule == "oracle":
            if r[k]["y"]:
                return k
            continue
        c0 = float(c[0].predict([r[k]["p"]])[0])
        if rule == "threshold":
            if c0 >= param:
                return k
            continue
        gains = []
        for h, v in r[k]["r"].items():
            if k + h <= r["ks"][-1] and h in c:
                ch = float(c[h].predict([v])[0])
                gains.append(max(ch, c0) - c0 - param * h)
        if not gains or max(gains) <= 0:
            return k
    return kk


def run_policy(rows, cals, rule, param, ntok):
    hit, npass, cost, toks = [], [], [], []
    for i, r in enumerate(rows):
        c = cals[i] if isinstance(cals, list) else cals
        kk = stop_depth(r, c, rule, param)
        s = r[kk]
        hit.append(s["y"]); npass.append(len(s["set"])); cost.append(s["cost"])
        toks.append(sum(ntok[r["id"]][j] for j in s["set"]))
    return {"FSR": 100 * float(np.mean(hit)), "passages": float(np.mean(npass)), "enc_seqs": float(np.mean(cost)),
            "reader_tokens": float(np.mean(toks))}


def token_counts(split):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QTOK)
    out = {}
    for e in load(f"{D}/{split}.jsonl"):
        out[e["id"]] = [len(tok(f"{c['title']}: {c['text']}", add_special_tokens=False)["input_ids"]) for c in e["ctxs"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--hold", required=True)
    ap.add_argument("--hold_split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write_adapt", default="", help="write the adaptive operating point (order = returned set) here")
    ap.add_argument("--match_k", type=int, default=4, help="operating point: cheapest lambda reaching fixed-k FSR on the HOLD-OUT")
    a = ap.parse_args()
    rows = features(a.res, a.split)
    hold = features(a.hold, a.hold_split)
    src = fit(hold)
    tgt = xfit(rows)
    ntok = token_counts(a.split)
    res = {"n": len(rows), "fixed": {k: run_policy(rows, src, "fixed", k, ntok) for k in range(1, K + 1)},
           "oracle": run_policy(rows, src, "oracle", 0, ntok), "source": {}, "target_xfit": {}}
    for name, cals in (("source", src), ("target_xfit", tgt)):
        res[name]["threshold"] = {t: run_policy(rows, cals, "threshold", t, ntok) for t in (0.3, 0.5, 0.7, 0.9)}
        res[name]["lookahead"] = {l: run_policy(rows, cals, "lookahead", l, ntok) for l in LAMS}
    # matched operating points: cheapest look-ahead point (source calibration) that reaches each fixed budget's FSR
    res["matched"] = {}
    for k in range(2, K + 1):
        f = res["fixed"][k]
        ok = [v for v in res["source"]["lookahead"].values() if v["FSR"] >= f["FSR"] - 1e-9]
        if ok:
            b = min(ok, key=lambda v: v["enc_seqs"])
            res["matched"][k] = {"fixed": f, "adaptive": b,
                                 "saving_passages_%": 100 * (1 - b["passages"] / f["passages"]),
                                 "saving_tokens_%": 100 * (1 - b["reader_tokens"] / f["reader_tokens"]),
                                 "saving_enc_%": 100 * (1 - b["enc_seqs"] / max(1, f["enc_seqs"]))}
    # calibration of the completeness map on the evaluated split: source-fitted (deployable) vs target cross-fitted
    def ece(cals):
        p, y = [], []
        for i, r in enumerate(rows):
            c = cals[i] if isinstance(cals, list) else cals
            for k in r["ks"]:
                if 2 <= k <= 4:
                    p.append(float(c[0].predict([r[k]["p"]])[0])); y.append(r[k]["y"])
        p, y = np.array(p), np.array(y)
        idx = np.minimum((p * 10).astype(int), 9)
        return 100 * float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(10) if (idx == b).any()))
    raw = np.array([r[k]["p"] for r in rows for k in r["ks"] if 2 <= k <= 4]); yy = np.array([r[k]["y"] for r in rows for k in r["ks"] if 2 <= k <= 4])
    ridx = np.minimum((raw * 10).astype(int), 9)
    res["ece"] = {"raw": 100 * float(sum((ridx == b).mean() * abs(raw[ridx == b].mean() - yy[ridx == b].mean()) for b in range(10) if (ridx == b).any())),
                  "source": ece(src), "target_xfit": ece(tgt)}
    # operating point chosen on the source hold-out only (cross-fitted there so that it is not fitted on itself)
    hx = xfit(hold)
    target = np.mean([r[a.match_k]["y"] for r in hold if a.match_k in r])
    lam_ok = [l for l in LAMS if np.mean([r[stop_depth(r, hx[i], "lookahead", l)]["y"] for i, r in enumerate(hold)]) >= target]
    lam = max(lam_ok) if lam_ok else min(LAMS)
    res["operating_point"] = {"lambda": lam, "matched_k": a.match_k, "hold_fixed_FSR": 100 * float(target),
                              "eval": run_policy(rows, src, "lookahead", lam, ntok)}
    if a.write_adapt:
        with open(a.write_adapt, "w") as f:
            for r in rows:
                kk = stop_depth(r, src, "lookahead", lam)
                f.write(json.dumps({"id": r["id"], "order": r[kk]["set"], "k": kk}) + chr(10))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("fixed", "oracle", "matched")}, indent=0)[:4000])


if __name__ == "__main__":
    main()
