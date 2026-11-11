"""Counterfactual diagnosis on gold proofs.

For every dev question and every g in G, the state S = G \\ {g} is fed to the model:
  * missing-evidence diagnosis: is g the arg-max proposal?  (Top-1 / MRR of g among candidates)
  * verdict discrimination (set heads): AUROC of the posterior verdict C(S) separating the complete set G
    from near-complete sets G \\ {g} and G \\ {g} + {strongest distractor}.
"""
import os
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from data import Pool, load_jsonl
from infer import run_model
from model import ERCNet

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    ck = torch.load(f"{a.ckpt}/model.pt", map_location="cpu")
    ar = ck["args"]
    tok = AutoTokenizer.from_pretrained(ar["enc"])
    head = "point_noset" if (ar["head"] == "point" and ar.get("no_set")) else ar["head"]
    model = ERCNet(ar["enc"], head=head, m=ar["m"], use_set=not ar.get("no_set")).cuda().eval()
    model.load_state_dict(ck["state"])
    ids_fn = (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    ns = ar.get("no_state", False)
    top1, rr, vy, vp = [], [], [], []
    for e in load_jsonl(f"{DATA}/{a.split}.jsonl", a.limit):
        p = Pool(e, tok)
        G, n = list(p.gold), len(p.d)
        if len(G) < 2:
            continue
        cand = list(range(n))
        states = [[x for x in G if x != g] for g in G]
        out = run_model(model, [(p, cand, st) for st in states] + [(p, cand, G)], ids_fn, "cuda", ns)
        score = out["logC_add"] if "logC_add" in out else F.logsigmoid(out["score"])
        hards = []
        for i, g in enumerate(G):
            s = score[i, :n].float().clone()
            s[states[i]] = -1e9
            rank = int((s > s[g]).sum()) + 1
            top1.append(float(rank == 1)); rr.append(1.0 / rank)
            if "logC" in out:
                vp.append(float(out["logC"][i])); vy.append(0)
                s[g] = -1e9
                hards.append(states[i] + [int(s.argmax())])  # strongest distractor completion
        if "logC" in out:
            vp.append(float(out["logC"][len(G)])); vy.append(1)
            out2 = run_model(model, [(p, cand, st) for st in hards], ids_fn, "cuda", ns)
            for i in range(len(hards)):  # posterior verdict after reading the wrong completion
                vp.append(float(out2["logC"][i])); vy.append(0)
    res = {"diag_top1": 100 * np.mean(top1), "diag_mrr": 100 * np.mean(rr), "n": len(top1)}
    if vy:
        res["verdict_auroc"] = 100 * roc_auc_score(vy, vp)
    print(res)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
