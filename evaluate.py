"""Metrics: full-support recall (FSR), support recall (SR), set-EM, token-budget FSR, QA EM/F1, calibration."""
import os
import collections
import json
import re
import string

import numpy as np

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


def load(p):
    return [json.loads(l) for l in open(p)]


def gold_map(split):
    return {e["id"]: e for e in load(f"{DATA}/{split}.jsonl")}


def fsr(order, gold, k):
    return float(set(gold).issubset(order[:k]))


def sr(order, gold, k):
    return len(set(gold) & set(order[:k])) / max(1, len(gold))


def retrieval_metrics(split, sel_path, ks=(2, 3, 4, 5, 6), key="order", budgets=(), by_hops=False):
    G = gold_map(split)
    rows = {r["id"]: r for r in load(sel_path)}
    res = collections.defaultdict(list)
    hop_res = collections.defaultdict(list)
    for qid, e in G.items():
        if qid not in rows or not e["gold"]:
            continue
        o = rows[qid][key]
        g = e["gold"]
        sets = rows[qid].get("sets")
        pick = (lambda k: sets.get(str(k), o)) if sets else (lambda k: o[:k])
        for k in ks:
            res[f"FSR@{k}"].append(fsr(pick(k), g, k))
            res[f"SR@{k}"].append(sr(pick(k), g, k))
            if by_hops:
                hop_res[(e["hops"], k)].append(fsr(pick(k), g, k))
        res["SetEM@|G|"].append(fsr(pick(len(g)), g, len(g)))
        for B in budgets:
            ob = rows[qid].get(f"tok{B}")
            if ob is not None:
                res[f"FSR@{B}tok"].append(float(set(g).issubset(ob)))
    out = {k: 100 * np.mean(v) for k, v in res.items()}
    if by_hops:
        out["by_hops"] = {f"{h}hop@{k}": 100 * np.mean(v) for (h, k), v in sorted(hop_res.items())}
    out["n"] = len(res["SetEM@|G|"])
    return out


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(p, g):
    pt, gt = norm(p).split(), norm(g).split()
    c = collections.Counter(pt) & collections.Counter(gt)
    ns = sum(c.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(pt), ns / len(gt)
    return 2 * pr * rc / (pr + rc)


def qa_metrics(split, pred_path):
    G = gold_map(split)
    em, ff = [], []
    for r in load(pred_path):
        ans = G[r["id"]]["answers"]
        em.append(max(float(norm(r["pred"]) == norm(a)) for a in ans))
        ff.append(max(f1(r["pred"], a) for a in ans))
    return {"EM": 100 * np.mean(em), "F1": 100 * np.mean(ff), "n": len(em)}


def calibration(split, sel_path, k, bins=10):
    """ECE / AUROC of exp(logC(S_k)) as a predictor of G subset of S_k."""
    from sklearn.metrics import roc_auc_score
    G = gold_map(split)
    p, y = [], []
    for r in load(sel_path):
        e = G[r["id"]]
        if "verdict" in r:
            if str(k) not in r["verdict"]:
                continue
            p.append(float(np.exp(r["verdict"][str(k)])))
            y.append(fsr(r["sets"][str(k)], e["gold"], k))
            continue
        if len(r["logC"]) < k:
            continue
        p.append(float(np.exp(r["logC"][k - 1])))
        y.append(fsr(r["order"], e["gold"], k))
    p, y = np.array(p), np.array(y)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"ECE": 100 * ece, "AUROC": 100 * roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan"),
            "mean_p": 100 * p.mean(), "acc": 100 * y.mean()}
