"""Can a signal steer the search?  Ranking of the depth-1 search states (single passages) by different readings.

For every question whose annotated passages are all in the pool, every candidate d is read once as [q; d] (for a
singleton the proposer's and the set reader's serialisations coincide) and ranked by
    prop : the proposal logit p(d | q, {})
    rung0: the completeness rung logit P(r({d}) = 0)          (all that a binary completeness verdict provides)
    rung1: the reachability rung logit P(r({d}) <= 1)
    er   : minus the expected deficit E[r({d})]                (the A* priority used by ProofGap)
We report how often the first-hop passage (gold[0]; MuSiQue/2Wiki annotate hop order) and any annotated passage are
among the top-2 (the lattice width) and top-4.

python depth1.py --out results_pg/depth1.json --split open_musique --ckpts pg=ckpt_pg/musique_pg bin=ckpt_pg/musique_pg_bin
"""
import argparse
import collections
import json
import os
import sys

import torch
from transformers import AutoTokenizer

from data import Pool, build_seqs, collate_sets, load_jsonl
from model import R_DEFICIT, ERCNet, resolve_enc

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


def load_model(path):
    ck = torch.load(f"{path}/model.pt", map_location="cpu"); ar = ck["args"]
    ar["enc"] = resolve_enc(ar["enc"])
    m = ERCNet(ar["enc"], head=ar["head"], use_set=False, n_rung=ar.get("rungs", R_DEFICIT)).cuda().eval()
    m.load_state_dict(ck["state"]); m.ladder_kind = ar.get("set_target", "ordinal")
    return m, AutoTokenizer.from_pretrained(ar["enc"])


@torch.no_grad()
def probe(m, tok, split, types=None):
    stat = collections.defaultdict(list)
    for e in load_jsonl(f"{DATA}/{split}.jsonl"):
        g = e["gold"]
        if not g or any(x >= len(e["ctxs"]) for x in g) or (types and e.get("type") not in types):
            continue
        p = Pool(e, tok); n = len(p.d)
        ids, att = collate_sets(build_seqs(p, list(range(n)), [], tok.cls_token_id, tok.sep_token_id), tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = m.encoder(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state[:, 0].float()
        u = m.ladder_from_h(h)
        crit = {"prop": m.cls(h).squeeze(-1), "rung0": u[:, 0], "rung1": u[:, 1], "er": -torch.sigmoid(-u[:, :-1]).sum(1)}
        for name, sc in crit.items():
            order = torch.argsort(sc.float(), descending=True).tolist()
            stat[f"{name}|top2_first"].append(g[0] in order[:2])
            stat[f"{name}|top2_any"].append(bool(set(order[:2]) & set(g)))
            stat[f"{name}|top4_first"].append(g[0] in order[:4])
    return {k: 100 * sum(v) / len(v) for k, v in stat.items()} | {"n": len(stat["er|top2_first"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--types", default="")
    ap.add_argument("--ckpts", nargs="+", required=True, help="label=checkpoint_dir")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.ckpts:
        label, path = spec.split("=", 1)
        m, tok = load_model(path)
        res[f"{a.split}|{label}"] = probe(m, tok, a.split, a.types.split(",") if a.types else None)
        print(label, a.split, res[f"{a.split}|{label}"], flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
