"""Measured retrieval cost on a random sample of questions.

Differences from latency2.py:
  * the questions are a uniform random sample of the split (latency2 took the first n, which on MuSiQue are all 2-hop);
  * the systems are interleaved question by question in a shuffled order, so load from other jobs on the same GPU
    falls on all of them alike;
  * Beam Retrieval is timed with a lean search that encodes only the unused candidates of every beam and makes no
    pass after the last hop (infer.search re-encodes the chosen passages and runs one unused pass).  The lean
    search is checked against infer.search on every question.

python latency3.py --split musique_dev --n 200 --pg CKPT --beam CKPT --out results_pg4/latency3.json
"""
import os
import argparse
import json
import math
import random
import statistics as st
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import data
import infer as I
from data import Pool, load_jsonl
from infer import focus_search, search
from latency import load

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


@torch.no_grad()
def beam_lean(model, pool, kmax, width, ids):
    """Beam Retrieval as published: at every hop each beam scores its unused candidates; sequences are ranked by
    the score of the current hop."""
    nodes, sets, n = [([], 0.0)], {}, len(pool.d)
    for depth in range(kmax):
        items, cls = [], []
        for s, _ in nodes:
            cl = [c for c in range(n) if c not in s]
            items.append((pool, cl, s)); cls.append(cl)
        if not items or not cls[0]:
            break
        out = I.run_model(model, items, ids, "cuda")
        children = {}
        for row, ((s, _), cl) in enumerate(zip(nodes, cls)):
            val = F.logsigmoid(out["score"][row, :len(cl)].float())
            for i in torch.topk(val, min(width, len(cl))).indices.tolist():
                key = frozenset(s + [cl[i]])
                if key not in children or float(val[i]) > children[key][1]:
                    children[key] = (s + [cl[i]], float(val[i]))
        nodes = sorted(children.values(), key=lambda x: -x[1])[:width]
        sets[depth + 1] = nodes[0][0]
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="musique_dev")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pg", required=True)
    ap.add_argument("--beam", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counter = {"seqs": 0}
    orig, orig_sets = I.collate, I.collate_sets

    def counting(items, *x):
        out = orig(items, *x)
        counter["seqs"] += out[0].shape[0]
        return out

    def counting_sets(seqs, pad):
        counter["seqs"] += len(seqs)
        return orig_sets(seqs, pad)
    I.collate, I.collate_sets = counting, counting_sets

    rng = random.Random(a.seed)
    allx = load_jsonl(f"{DATA}/{a.split}.jsonl")
    exs = rng.sample(allx, a.n)
    hops = {}
    for e in exs:
        hops[len(e["gold"])] = hops.get(len(e["gold"]), 0) + 1
    res = {"split": a.split, "n": len(exs), "seed": a.seed, "gpu": torch.cuda.get_device_name(0), "supports": hops}
    print("sample", res, flush=True)

    from sentence_transformers import CrossEncoder
    rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, automodel_args={"torch_dtype": torch.float16})
    pairs = [[(e["question"], f"{c['title']}. {c['text']}") for c in e["ctxs"]] for e in exs]
    mb, arb = load(a.beam)
    cmb = arb.get("ctx_max") or data.SMAX
    mp, arp = load(a.pg)
    cmp_ = arp.get("ctx_max") or data.SMAX
    assert arb["enc"] == arp["enc"], (arb["enc"], arp["enc"])
    tok = AutoTokenizer.from_pretrained(arp["enc"])
    pools = [Pool(e, tok) for e in exs]
    ids = (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)

    def run_rr(i, K):
        rr.predict(pairs[i], batch_size=32)
        counter["seqs"] = len(pairs[i])

    def run_beam(i, K):
        data.set_ctx_max(cmb)
        return beam_lean(mb, pools[i], K, 2, ids)

    def run_pg(i, K):
        data.set_ctx_max(cmp_)
        return focus_search(mp, [pools[i]], K, 2, 4, "cuda", ids, 1, front="pooled")

    # untimed: warm-up, the stop depth of every question, and the check of the lean beam against infer.search
    for i in range(3):
        run_rr(i, 0); run_beam(i, 4); run_pg(i, 4)
    depth, mismatch, complete = [], 0, 0
    for i, p in enumerate(pools):
        r = run_pg(i, 5)[0]
        d = 5
        for k in range(1, 6):
            if str(k) in r["ladder"] and 1 / (1 + math.exp(-r["ladder"][str(k)][0])) >= 0.5:
                d = k
                break
        depth.append(d)
        complete += set(p.gold_all) <= set(r["sets"][str(d)])
        data.set_ctx_max(cmb)
        ref = search(mb, [p], 4, 2, "cuda", ids, 1, accum=False)[0]["sets"]
        lean = beam_lean(mb, p, 4, 2, ids)
        mismatch += any(sorted(ref[str(k)]) != sorted(lean[k]) for k in (2, 3, 4))
    res["beam_lean_differs_from_infer_search"] = mismatch
    res["stop_complete_pct"] = 100 * complete / len(pools)
    print("stop depth mean", sum(depth) / len(depth), "complete", res["stop_complete_pct"], "lean beam mismatches", mismatch, flush=True)

    cfg = [("bge-reranker-v2-m3", run_rr, 0)]
    cfg += [(f"Beam Retrieval K={K}", run_beam, K) for K in (2, 3, 4)]
    cfg += [(f"ProofGap K={K}", run_pg, K) for K in (2, 3, 4)]
    cfg += [("ProofGap stop", run_pg, None)]
    acc = {name: {"ms": [], "seqs": []} for name, _, _ in cfg}
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(pools)):
        order = list(cfg)
        rng.shuffle(order)
        for name, fn, K in order:
            counter["seqs"] = 0
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(i, depth[i] if K is None else K)
            torch.cuda.synchronize()
            acc[name]["ms"].append(1000 * (time.perf_counter() - t0))
            acc[name]["seqs"].append(counter["seqs"])
        if (i + 1) % 20 == 0:
            print(i + 1, {n: round(st.mean(v["ms"])) for n, v in acc.items()}, flush=True)
    for name, v in acc.items():
        res[name] = {"ms_per_q": st.mean(v["ms"]), "ms_median": st.median(v["ms"]), "seqs_per_q": st.mean(v["seqs"])}
    res["ProofGap stop"]["mean_depth"] = sum(depth) / len(depth)
    res["peak_mem_gb_all_models_loaded"] = torch.cuda.max_memory_allocated() / 2 ** 30
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
