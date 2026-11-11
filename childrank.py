"""How well does each reading rank the children of a search state?  (design check of the incremental reading)

For every evaluated question and three kinds of states S (empty, hop-ordered prefix, leave-one-out), every candidate
d is encoded once as [q; S; d]; the children are ranked by
    prop : the proposal logit
    er   : minus the incremental expected deficit E[r(S + {d})]
and we report hit@1 / hit@2 of a still-missing annotated passage (in the pool).

python childrank.py --ckpt ckpt_pg/musique_pg --split musique_dev --limit 600
"""
import os
import argparse
import collections
import random

import torch

from data import Pool, collate, load_jsonl
from depth1 import load_model

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()
    m, tok = load_model(a.ckpt)
    rng = random.Random(0)
    stat = collections.defaultdict(list)
    for e in load_jsonl(f"{DATA}/{a.split}.jsonl", a.limit):
        p = Pool(e, tok)
        g = p.gold
        if len(g) < 2:
            continue
        states = {"empty": [], "prefix": g[: rng.randrange(1, len(g))], "loo": [x for x in g if x != rng.choice(g)]}
        for name, st in states.items():
            miss = set(g) - set(st)
            cand = list(range(len(p.d)))
            ids, attn, mask, sel = collate([(p, cand, st)], tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(ids.cuda(), attn.cuda(), mask.cuda(), sel.cuda())
            u = m.ladder_from_h(out["h"][0])
            crit = {"prop": out["score"][0].float(), "er": -torch.sigmoid(-u[:, :-1]).sum(1)}
            for c, sc in crit.items():
                sc = sc.clone(); sc[st] = -1e9
                order = torch.argsort(sc, descending=True).tolist()
                stat[f"{name}|{c}|hit1"].append(order[0] in miss)
                stat[f"{name}|{c}|hit2"].append(bool(set(order[:2]) & miss))
    for k in sorted(stat):
        print(f"{k:22s} {100 * sum(stat[k]) / len(stat[k]):.1f}  (n={len(stat[k])})")


if __name__ == "__main__":
    main()
