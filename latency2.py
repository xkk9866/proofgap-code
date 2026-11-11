"""Measured retrieval cost: wall-clock per question, encoder sequences per question, peak memory (loading excluded).

Every system is timed on the same questions, one after the other on the same GPU, one question at a time.

python latency2.py --split musique_dev --n 200 --pg CKPT --beam CKPT --out results_pg4/latency2.json
"""
import os
import argparse
import json
import math
import time

import torch
from transformers import AutoTokenizer

import data
import infer as I
from data import Pool, load_jsonl
from latency import load
from infer import focus_search, search

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


def timed(fn, pools, counter):
    counter["seqs"] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = fn(pools)
    torch.cuda.synchronize()
    return out, 1000 * (time.time() - t0) / len(pools), counter["seqs"] / len(pools)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="musique_dev")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pg", required=True)
    ap.add_argument("--beam", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counter = {"seqs": 0}
    orig, orig_sets = data.collate, I.collate_sets

    def counting(items, *x):
        out = orig(items, *x)
        counter["seqs"] += out[0].shape[0]
        return out

    def counting_sets(seqs, pad):
        counter["seqs"] += len(seqs)
        return orig_sets(seqs, pad)
    I.collate, I.collate_sets = counting, counting_sets
    exs = load_jsonl(f"{DATA}/{a.split}.jsonl", a.n)
    res = {"split": a.split, "n": len(exs), "gpu": torch.cuda.get_device_name(0)}

    # ---- cross-encoder reranker: one pass over the pool
    from sentence_transformers import CrossEncoder
    rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, automodel_args={"torch_dtype": torch.float16})
    pairs = [[(e["question"], f"{c['title']}. {c['text']}") for c in e["ctxs"]] for e in exs]
    rr.predict(pairs[0], batch_size=32)
    torch.cuda.synchronize(); t0 = time.time()
    for p in pairs:
        rr.predict(p, batch_size=32)
    torch.cuda.synchronize()
    res["bge-reranker-v2-m3"] = {"ms_per_q": 1000 * (time.time() - t0) / len(exs), "seqs_per_q": sum(len(p) for p in pairs) / len(exs)}
    print("bge-reranker-v2-m3", res["bge-reranker-v2-m3"], flush=True)
    del rr; torch.cuda.empty_cache()

    # ---- Beam Retrieval and ProofGap: the same backbone, one question at a time
    for name, ck, mk in (("Beam Retrieval", a.beam, lambda m, ids, K: (lambda P: search(m, P, K, 2, "cuda", ids, 1, accum=False))),
                         ("ProofGap", a.pg, lambda m, ids, K: (lambda P: focus_search(m, P, K, 2, 4, "cuda", ids, 1, front="pooled")))):
        m, ar = load(ck)
        tok = AutoTokenizer.from_pretrained(ar["enc"])
        pools = [Pool(e, tok) for e in exs]
        ids = (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
        mk(m, ids, 5)(pools[:4])  # warm-up
        torch.cuda.reset_peak_memory_stats()
        for K in (2, 3, 4, 5):
            out, ms, sq = timed(mk(m, ids, K), pools, counter)
            res[f"{name} K={K}"] = {"ms_per_q": ms, "seqs_per_q": sq}
            print(name, K, res[f"{name} K={K}"], flush=True)
        if name == "ProofGap":  # the stop rule: every question is searched to the depth at which P(r = 0) >= 1/2
            depth = []
            for r in out:
                d = 5
                for k in range(1, 6):
                    if str(k) in r["ladder"] and 1 / (1 + math.exp(-r["ladder"][str(k)][0])) >= 0.5:
                        d = k
                        break
                depth.append(d)
            torch.cuda.synchronize(); t0 = time.time(); counter["seqs"] = 0
            for p, d in zip(pools, depth):
                focus_search(m, [p], d, 2, 4, "cuda", ids, 1, front="pooled")
            torch.cuda.synchronize()
            res["ProofGap stop"] = {"ms_per_q": 1000 * (time.time() - t0) / len(pools), "seqs_per_q": counter["seqs"] / len(pools),
                                    "mean_depth": sum(depth) / len(depth)}
            print("ProofGap stop", res["ProofGap stop"], flush=True)
        res[f"{name} peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        del m; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
