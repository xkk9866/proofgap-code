"""Wall-clock latency per question and encoder sequences per question (model loading excluded).

python latency.py --n 300 --out results/latency.json
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

import data
from data import Pool, load_jsonl
from infer import focus_search, proof_search, prover_search, search
from model import ERCNet, resolve_enc

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
C = "/data/yaokang/ercrag/ckpt"


def load(ck):
    ck = torch.load(f"{ck}/model.pt", map_location="cpu")
    ar = ck["args"]
    ar["enc"] = resolve_enc(ar["enc"])
    data.set_ctx_max(ar.get("ctx_max") or data.SMAX)  # each model reads context passages as it was trained to
    head = "point_noset" if (ar["head"] == "point" and ar.get("no_set")) else ar["head"]
    m = ERCNet(ar["enc"], head=head, m=ar["m"], use_set=not ar.get("no_set"), hop_heads=ar.get("hop_heads", False),
               n_rung=ar.get("rungs", 4)).cuda().eval()
    m.load_state_dict(ck["state"])
    m.sep_id = 2
    m.ladder_kind = ar.get("set_target", "ordinal")
    return m, ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="musique_dev_s600")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam_ckpt", default="musique_beam")
    ap.add_argument("--pg_ckpt", default="")
    ap.add_argument("--pg4_ckpt", default="", help="ProofGap v4 checkpoint (member-focused consensus search)")
    ap.add_argument("--pg5_ckpt", default="", help="ProofGap v5 checkpoint (membership proposals + pooled-deficit frontier)")
    ap.add_argument("--pg6_ckpt", default="", help="ProofGap v6 checkpoint (deficit and necessity readings, exchange move)")
    ap.add_argument("--ckroot", default=C)
    a = ap.parse_args()
    selection = os.path.join(os.environ.get('RES', os.path.dirname(a.out)), 'g1_selection.json')
    cand = json.load(open(selection))['candidate'] if os.path.exists(selection) else 'ladder'
    counter = {"seqs": 0}
    orig = data.collate
    import infer as I

    def counting(items, *x):
        out = orig(items, *x)
        counter["seqs"] += out[0].shape[0]
        return out
    I.collate = counting
    orig_sets = I.collate_sets

    def counting_sets(seqs, pad):
        counter["seqs"] += len(seqs)
        return orig_sets(seqs, pad)
    I.collate_sets = counting_sets
    res = {}
    jobs = [
        ("Beam b=1", a.beam_ckpt, lambda m, P, ids: search(m, P, 5, 1, "cuda", ids, 1, accum=False)),
        ("Beam b=2", a.beam_ckpt, lambda m, P, ids: search(m, P, 5, 2, "cuda", ids, 1, accum=False)),
    ]
    if a.pg6_ckpt:
        jobs += [("ProofGap W=2", a.pg6_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, front="pooled")),
                 ("ProofGap W=2 exchange", a.pg6_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, front="pooled", exchange=True)),
                 ("ProofGap W=4", a.pg6_ckpt, lambda m, P, ids: focus_search(m, P, 5, 4, 4, "cuda", ids, 1, front="pooled"))]
    elif a.pg5_ckpt:  # ProofGap v5: membership proposals, pooled-deficit frontier, member-view consensus (+ controls)
        jobs += [("ProofGap W=2", a.pg5_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, cand="member", front="pooled")),
                 ("ProofGap W=2 (holistic reading)", a.pg5_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, cand="member", front="pooled", views="holistic")),
                 ("ProofGap W=2 (chain lattice)", a.pg5_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, cand="member", front="chain")),
                 ("ProofGap W=4", a.pg5_ckpt, lambda m, P, ids: focus_search(m, P, 5, 4, 4, "cuda", ids, 1, cand="member", front="pooled"))]
    elif a.pg4_ckpt:  # ProofGap v4: member-focused consensus search (+ its holistic-reading and chain-lattice controls)
        jobs += [("ProofGap W=2", a.pg4_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, front="pooled")),
                 ("ProofGap W=2 (holistic reading)", a.pg4_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, views="holistic", front="pooled")),
                 ("ProofGap W=2 (chain lattice)", a.pg4_ckpt, lambda m, P, ids: focus_search(m, P, 5, 2, 4, "cuda", ids, 1, cand="chain", front="chain")),
                 ("ProofGap W=4", a.pg4_ckpt, lambda m, P, ids: focus_search(m, P, 5, 4, 4, "cuda", ids, 1, front="pooled"))]
    elif a.pg_ckpt:  # ProofGap v3: deficit-guided proof search (and the v2 chain lattice on the same network)
        jobs += [("ProofGap W=2", a.pg_ckpt, lambda m, P, ids: proof_search(m, P, 5, 2, 4, "cuda", ids, 2, cand=cand)),
                 ("ProofGap W=2 (chain lattice)", a.pg_ckpt, lambda m, P, ids: proof_search(m, P, 5, 2, 4, "cuda", ids, 2, cand="chain", front="chain")),
                 ("ProofGap W=4", a.pg_ckpt, lambda m, P, ids: proof_search(m, P, 5, 4, 4, "cuda", ids, 2, cand=cand))]
    else:
        jobs += [("PROVER W=2", "musique_prover", lambda m, P, ids: prover_search(m, P, 5, 2, 4, "cuda", ids, 2)),
                 ("ProofGap W=2", "musique_def", lambda m, P, ids: prover_search(m, P, 5, 2, 4, "cuda", ids, 2, frontier="deficit", decide="or"))]
    for name, ck, fn in jobs:
        m, ar = load(f"{a.ckroot}/{ck}")
        tok = AutoTokenizer.from_pretrained(ar["enc"])
        pools = [Pool(e, tok) for e in load_jsonl(f"{DATA}/{a.split}.jsonl", a.n)]
        ids = (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
        fn(m, pools[:4], ids)  # warm-up
        counter["seqs"] = 0
        torch.cuda.synchronize(); t0 = time.time()
        fn(m, pools, ids)
        torch.cuda.synchronize(); dt = time.time() - t0
        res[name] = {"ms_per_q": 1000 * dt / len(pools), "seqs_per_q": counter["seqs"] / len(pools),
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        print(name, res[name], flush=True)
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
