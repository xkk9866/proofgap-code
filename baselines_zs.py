"""Zero-shot ranking baselines over a candidate pool: BM25, bge-large dense, bge-reranker-v2-m3, MMR.

Writes one jsonl per method with full orderings ({id, order}); budgets are applied at evaluation.
"""
import os
import argparse
import json
import re

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from data import ctx_str, load_jsonl

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
RES = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")
QINST = "Represent this sentence for searching relevant passages: "


def toks(s):
    return re.findall(r"\w+", s.lower())


def mmr(rel, emb, lam=0.7):
    n = len(rel)
    r = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    sim = emb @ emb.T
    order, left = [], list(range(n))
    while left:
        if not order:
            best = max(left, key=lambda i: r[i])
        else:
            best = max(left, key=lambda i: lam * r[i] - (1 - lam) * max(sim[i, j] for j in order))
        order.append(best); left.remove(best)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    exs = load_jsonl(f"{DATA}/{a.split}.jsonl", a.limit)
    from sentence_transformers import SentenceTransformer, CrossEncoder
    emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    emb_model.half()
    rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, automodel_args={"torch_dtype": torch.float16})
    outs = {k: open(f"{RES}/{a.split}.{k}.jsonl", "w") for k in ["bm25", "dense", "rerank", "mmr"]}
    for i, e in enumerate(exs):
        docs = [ctx_str(c) for c in e["ctxs"]]
        bm = BM25Okapi([toks(d) for d in docs]).get_scores(toks(e["question"]))
        qv = emb_model.encode([QINST + e["question"]], normalize_embeddings=True)
        dv = emb_model.encode(docs, normalize_embeddings=True, batch_size=64)
        dn = (dv @ qv[0])
        rs = np.array(rr.predict([(e["question"], d) for d in docs], batch_size=64))
        rows = {"bm25": list(np.argsort(-bm)), "dense": list(np.argsort(-dn)), "rerank": list(np.argsort(-rs)),
                "mmr": mmr(rs, dv)}
        for k, o in rows.items():
            outs[k].write(json.dumps({"id": e["id"], "order": [int(x) for x in o]}) + "\n")
        if i % 500 == 0:
            print(i, flush=True)
    for f in outs.values():
        f.close()


if __name__ == "__main__":
    main()
