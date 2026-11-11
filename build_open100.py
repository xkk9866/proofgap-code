"""Open-corpus pools with a 100-passage first stage (open100_{ds}.jsonl), from the bge-large embeddings cached by
est_open.py.  Same format as open_prep.py: gold passages outside the pool get ids >= 1000 and count as failures."""
import json
import sys

import numpy as np

from est_open import EMB
from open_prep import OUT, RAW, key, supports

TOPN = 100


def main(ds):
    name = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}[ds]
    qs = json.load(open(f"{RAW}/{name}.json"))
    corpus = json.load(open(f"{RAW}/{name}_corpus.json"))
    cidx = {key(c["title"], c["text"]): i for i, c in enumerate(corpus)}
    by_title = {}
    for i, c in enumerate(corpus):
        by_title.setdefault(c["title"].strip(), []).append(i)
    D, Q = np.load(f"{EMB}/{ds}_corpus.npy"), np.load(f"{EMB}/{ds}_q.npy")
    S = Q.astype(np.float32) @ D.astype(np.float32).T
    old = {json.loads(l)["id"]: json.loads(l) for l in open(f"{OUT}/open_{ds}.jsonl")}
    rows, same30 = [], []
    for qi, e in enumerate(qs):
        sup, ans = supports(e)
        gids = []
        for t, x in sup:
            k = key(t, x)
            if k in cidx:
                gids.append(cidx[k])
            elif t.strip() in by_title:
                gids.append(by_title[t.strip()][0])
        top = list(np.argsort(-S[qi])[:TOPN])
        gold = [top.index(g) if g in top else 1000 + j for j, g in enumerate(gids)]
        qid = str(e.get("id", e.get("_id", qi)))
        if qid in old:
            o = old[qid]["ctxs"]
            same30.append(len({c["title"] + c["text"][:50] for c in o} & {corpus[i]["title"] + corpus[i]["text"][:50] for i in top[:30]}) / max(1, len(o)))
        rows.append({"id": qid, "question": e["question"], "answers": ans,
                     "ctxs": [{"title": corpus[i]["title"], "text": corpus[i]["text"]} for i in top], "gold": gold,
                     "hops": len(gids), "type": str(e.get("id", e.get("type", "")))})
    with open(f"{OUT}/open100_{ds}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    inpool = np.mean([all(g < 1000 for g in r["gold"]) for r in rows])
    print(f"{ds}: wrote {len(rows)}; all-gold-in-top{TOPN} {100 * inpool:.1f}%; top-30 overlap with open_{ds} {100 * np.mean(same30):.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
