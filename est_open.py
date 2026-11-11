"""Coverage of the open-corpus pools: all-gold-in-top-n for the question query, and whether a frontier query
[q; passages already held] retrieves the passages still missing (oracle: held = gold passages in the top-30).

python est_open.py musique|2wiki|hotpotqa   -> also caches corpus / question embeddings under data/emb/
"""
import json
import os
import sys

import numpy as np

from open_prep import QINST, RAW, key, supports

EMB = "/data/yaokang/ercrag/data/emb"


def main(ds):
    name = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}[ds]
    qs = json.load(open(f"{RAW}/{name}.json"))
    corpus = json.load(open(f"{RAW}/{name}_corpus.json"))
    cidx = {key(c["title"], c["text"]): i for i, c in enumerate(corpus)}
    by_title = {}
    for i, c in enumerate(corpus):
        by_title.setdefault(c["title"].strip(), []).append(i)
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer("BAAI/bge-large-en-v1.5", device=dev)
    if dev == "cuda":
        m.half()
    os.makedirs(EMB, exist_ok=True)
    pD = f"{EMB}/{ds}_corpus.npy"
    if os.path.exists(pD):
        D = np.load(pD)
    else:
        D = m.encode([f"{c['title']}: {c['text']}" for c in corpus], batch_size=128, normalize_embeddings=True)
        np.save(pD, D)
    Q = m.encode([QINST + e["question"] for e in qs], batch_size=128, normalize_embeddings=True)
    np.save(f"{EMB}/{ds}_q.npy", Q)
    S = Q @ D.T
    G = []
    for e in qs:
        sup, _ = supports(e)
        g = []
        for t, x in sup:
            k = key(t, x)
            if k in cidx:
                g.append(cidx[k])
            elif t.strip() in by_title:
                g.append(by_title[t.strip()][0])
        G.append(g)
    order = np.argsort(-S, axis=1)
    for n in (30, 50, 100, 200):
        top = order[:, :n]
        print(f"{ds} all-gold-in-top{n}: {100 * np.mean([all(x in set(top[i]) for x in G[i]) for i in range(len(qs))]):.1f}%")
    # frontier query: question + gold passages already in the top-30 (as text), retrieve the missing ones
    fq, idx = [], []
    for i, e in enumerate(qs):
        top30 = set(order[i, :30].tolist())
        held = [x for x in G[i] if x in top30]
        miss = [x for x in G[i] if x not in top30]
        if miss and held:
            txt = " ".join(f"{corpus[x]['title']}: {corpus[x]['text']}" for x in held)
            fq.append(QINST + e["question"] + " " + txt); idx.append((i, held, miss))
    F = m.encode(fq, batch_size=64, normalize_embeddings=True)
    SF = F @ D.T
    for n in (5, 10, 20):
        rec, grow = [], []
        for j, (i, held, miss) in enumerate(idx):
            top = set(np.argsort(-SF[j])[:n].tolist())
            rec.append(all(x in top for x in miss))
        print(f"{ds} frontier query top{n}: missing gold recovered for {100 * np.mean(rec):.1f}% of the {len(idx)} partially covered "
              f"questions -> ceiling {100 * (np.mean([all(x in set(order[i, :30]) for x in G[i]) for i in range(len(qs))]) + np.sum(rec) / len(qs)):.1f}%")
    no_held = sum(1 for i in range(len(qs)) if G[i] and not any(x in set(order[i, :30]) for x in G[i]))
    print(f"{ds} questions with no gold in top-30: {100 * no_held / len(qs):.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
