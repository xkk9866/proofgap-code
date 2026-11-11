"""Open-style training pools: the HippoRAG open-corpus construction applied to TRAINING questions only.

Training questions are cut into blocks of 1,000 (file order).  The corpus of a block is the deduplicated union of
the candidate paragraphs (gold and distractors) of its questions -- exactly how the 1,000-question open benchmarks
of HippoRAG build their corpora from dev questions -- so distractors are topically close passages written for
*other* questions and some annotated passages are not retrieved at all.  bge-large-en-v1.5 (same query instruction
as open_prep.py) retrieves the top-30 paragraphs of the block corpus for every question of the block; annotated
passages outside the top-30 get ids >= 1000 and count as missing requirements (data.Pool.n_out).

Writes to data/proc:
  {ds}_train_open.jsonl            aligned with {ds}_train.jsonl (same ids and questions; only the pool differs)
  {ds}_hold.jsonl, {ds}_hold_open.jsonl
                                   the last H training questions (excluded by train.py --holdout H); used to fit the
                                   calibration maps of the stopping rule and nothing else (no dev/test labels).

python build_open_train.py musique 19938 1000
python build_open_train.py hotpotqa 21000 1000
python build_open_train.py 2wiki 21000 1000
"""
import json
import sys

import numpy as np

OUT = "/data/yaokang/ercrag/data/proc"
QINST = "Represent this sentence for searching relevant passages: "
TOPN, BLOCK = 30, 1000


def key(c):
    return (c["title"].strip(), " ".join(c["text"].split())[:200])


def main(ds, limit, hold):
    exs = []
    with open(f"{OUT}/{ds}_train.jsonl") as f:
        for line in f:
            exs.append(json.loads(line))
            if len(exs) >= limit:
                break
    exs = [e for e in exs if len(e["gold"]) > 0]  # identical filtering to train.py
    uniq, idx = [], {}
    for e in exs:
        for c in e["ctxs"]:
            k = key(c)
            if k not in idx:
                idx[k] = len(uniq); uniq.append(c)
    print(ds, len(exs), "questions,", len(uniq), "unique paragraphs", flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    m = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    m.half()
    D = m.encode([f"{c['title']}: {c['text']}" for c in uniq], batch_size=64, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    Q = m.encode([QINST + e["question"] for e in exs], batch_size=64, normalize_embeddings=True,
                 convert_to_numpy=True).astype(np.float32)
    rows, inpool = [], []
    for b0 in range(0, len(exs), BLOCK):
        block = exs[b0: b0 + BLOCK]
        cids = sorted({idx[key(c)] for e in block for c in e["ctxs"]})
        Db = D[cids]
        S = Q[b0: b0 + len(block)] @ Db.T
        for qi, e in enumerate(block):
            top = [cids[j] for j in np.argsort(-S[qi])[:TOPN]]
            pos = {c: i for i, c in enumerate(top)}
            gids = [idx[key(e["ctxs"][g])] for g in e["gold"]]
            gold = [pos[g] if g in pos else 1000 + j for j, g in enumerate(gids)]
            inpool.append(all(g < 1000 for g in gold))
            rows.append({"id": e["id"], "question": e["question"], "answers": e["answers"],
                         "ctxs": [uniq[c] for c in top], "gold": gold, "hops": e.get("hops", len(gold)),
                         "type": e.get("type", "")})
    with open(f"{OUT}/{ds}_train_open.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/{ds}_hold.jsonl", "w") as f:
        for e in exs[len(exs) - hold:]:
            f.write(json.dumps(e) + "\n")
    with open(f"{OUT}/{ds}_hold_open.jsonl", "w") as f:
        for r in rows[len(rows) - hold:]:
            f.write(json.dumps(r) + "\n")
    print(f"{ds}: {len(rows)} open-style pools; all gold in top-{TOPN}: {100*np.mean(inpool):.1f}%; hold-out {hold}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
