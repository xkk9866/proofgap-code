"""Unify MuSiQue / HotpotQA / 2WikiMultiHopQA into {id, question, answers, ctxs, gold, hops, type} jsonl."""
import os
import json
import random
import sys

import pandas as pd

RAW = os.environ.get("PROOFGAP_RAW_DIR", "data/raw")
OUT = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")


def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows))


def musique(split):
    rows = []
    for line in open(f"{RAW}/MuSiQue/musique_ans_v1.0_{split}.jsonl"):
        e = json.loads(line)
        if not e.get("answerable", True):
            continue
        ctxs = [{"title": p["title"], "text": p["paragraph_text"]} for p in e["paragraphs"]]
        gold = []
        for d in e["question_decomposition"]:
            i = d["paragraph_support_idx"]
            if i is not None and i not in gold:
                gold.append(i)
        for p in e["paragraphs"]:
            if p["is_supporting"] and p["idx"] not in gold:
                gold.append(p["idx"])
        rows.append({"id": e["id"], "question": e["question"], "answers": [e["answer"]] + e.get("answer_aliases", []),
                     "ctxs": ctxs, "gold": gold, "hops": len(e["question_decomposition"]), "type": e["id"].split("__")[0]})
    return rows


def hotpot(split):
    files = {"train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
             "dev": ["validation-00000-of-00001.parquet"]}[split]
    df = pd.concat([pd.read_parquet(f"{RAW}/hotpot_qa/distractor/{f}") for f in files])
    rows = []
    for e in df.itertuples():
        titles = list(e.context["title"])
        ctxs = [{"title": t, "text": "".join(s)} for t, s in zip(titles, e.context["sentences"])]
        gold = []
        for t in e.supporting_facts["title"]:
            if t in titles and titles.index(t) not in gold:
                gold.append(titles.index(t))
        rows.append({"id": e.id, "question": e.question, "answers": [e.answer], "ctxs": ctxs, "gold": gold,
                     "hops": 2, "type": e.type})
    return rows


def twowiki(split):
    import glob
    fs = sorted(glob.glob(f"{RAW}/2WikiMultihopQA/**/*{ {'train': 'train', 'dev': 'dev'}[split] }*", recursive=True))
    fs = [f for f in fs if f.endswith((".parquet", ".json", ".jsonl"))]
    if not fs and split == "dev":
        fs = [f for f in glob.glob(f"{RAW}/2WikiMultihopQA/**/*valid*", recursive=True) if f.endswith((".parquet", ".json", ".jsonl"))]
    print(fs)
    df = pd.concat([pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=f.endswith(".jsonl")) for f in fs])
    rows = []
    for e in df.to_dict("records"):
        ctx = e["context"]
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        if isinstance(ctx, dict):
            titles, sents = list(ctx["title"]), list(ctx["sentences"] if "sentences" in ctx else ctx["content"])
        else:
            titles, sents = [c[0] for c in ctx], [c[1] for c in ctx]
        ctxs = [{"title": t, "text": " ".join(s) if not isinstance(s, str) else s} for t, s in zip(titles, sents)]
        sf = e["supporting_facts"]
        if isinstance(sf, str):
            sf = json.loads(sf)
        sft = list(sf["title"]) if isinstance(sf, dict) else [x[0] for x in sf]
        gold = []
        for t in sft:
            if t in titles and titles.index(t) not in gold:
                gold.append(titles.index(t))
        qid = e.get("_id", e.get("id"))
        rows.append({"id": qid, "question": e["question"], "answers": [e["answer"]], "ctxs": ctxs, "gold": gold,
                     "hops": len(gold), "type": e.get("type", "")})
    return rows


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    ds = sys.argv[1]
    rng = random.Random(0)
    fn = {"musique": musique, "hotpotqa": hotpot, "2wiki": twowiki}[ds]
    tr, dv = fn("train"), fn("dev")
    if len(tr) > 30000:
        tr = rng.sample(tr, 30000)
    dump(tr, f"{OUT}/{ds}_train.jsonl")
    dump(dv, f"{OUT}/{ds}_devfull.jsonl")
    if len(dv) > 3000:
        dv = random.Random(1).sample(dv, 3000)
    dump(dv, f"{OUT}/{ds}_dev.jsonl")
