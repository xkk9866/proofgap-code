"""Open-corpus setting (HippoRAG 1,000-question benchmark): bge-large-en-v1.5 retrieves top-30 from the merged
corpus; methods then select budgeted evidence sets from that pool.  Gold passages missing from the pool are
encoded with ids >= 1000 so that they can never be selected (they count as failures for FSR/SR)."""
import json
import sys

import numpy as np

RAW = "/data/yaokang/ercrag/data/HippoRAG_2"
OUT = "/data/yaokang/ercrag/data/proc"
QINST = "Represent this sentence for searching relevant passages: "
TOPN = 30


def key(t, x):
    return (t.strip(), " ".join(x.split())[:200])


def supports(e):
    """Return list of (title, text) supporting passages and answers in any of the three raw formats."""
    if "paragraphs" in e:
        sup = [(p["title"], p.get("paragraph_text", p.get("text", ""))) for p in e["paragraphs"] if p.get("is_supporting")]
        ans = [e["answer"]] + e.get("answer_aliases", [])
        return sup, ans
    ctx = {c[0]: (" ".join(c[1]) if isinstance(c[1], list) else c[1]) for c in e["context"]}
    titles = []
    for t, _ in e["supporting_facts"]:
        if t not in titles:
            titles.append(t)
    return [(t, ctx[t]) for t in titles if t in ctx], [e["answer"]]


def main(ds):
    name = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}[ds]
    qs = json.load(open(f"{RAW}/{name}.json"))
    corpus = json.load(open(f"{RAW}/{name}_corpus.json"))
    print(ds, len(qs), len(corpus), list(qs[0].keys()), list(corpus[0].keys()))
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
    D = m.encode([f"{c['title']}: {c['text']}" for c in corpus], batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    Q = m.encode([QINST + e["question"] for e in qs], batch_size=128, normalize_embeddings=True)
    S = Q @ D.T
    rows, miss = [], 0
    for qi, e in enumerate(qs):
        sup, ans = supports(e)
        gids = []
        for t, x in sup:
            k = key(t, x)
            if k in cidx:
                gids.append(cidx[k])
            elif t.strip() in by_title:
                gids.append(by_title[t.strip()][0])
            else:
                miss += 1
        top = list(np.argsort(-S[qi])[:TOPN])
        gold = [top.index(g) if g in top else 1000 + j for j, g in enumerate(gids)]
        rows.append({"id": str(e.get("id", e.get("_id", qi))), "question": e["question"], "answers": ans,
                     "ctxs": [{"title": corpus[i]["title"], "text": corpus[i]["text"]} for i in top], "gold": gold,
                     "hops": len(gids), "type": str(e.get("id", e.get("type", "")))})
    with open(f"{OUT}/open_{ds}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    inpool = np.mean([all(g < 1000 for g in r["gold"]) for r in rows])
    print(f"{ds}: wrote {len(rows)}; unmatched supports {miss}; all-gold-in-top{TOPN} {100*inpool:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
