"""Control for the stop rule: the length of the proof predicted from the question alone.

A sequence classifier (the backbone of the retrievers) reads the question and predicts the number of annotated
supporting passages, in the manner of a query-complexity classifier.  A retriever that returns its set of that
predicted length is the alternative to stopping by a reading of the evidence.

python lenclf.py --data musique --seed 0 --out /data/yaokang/ercrag/results_pg4/lenclf --splits musique_dev open_musique
writes {out}.{split}.s{seed}.json  {question id: predicted number of supports}
"""
import os
import argparse
import collections
import json
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import load_jsonl
from model import resolve_enc

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="musique")
    ap.add_argument("--enc", default="/data/yaokang/models/dv3/deberta-v3-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--holdout", type=int, default=1000)  # the same questions the retrievers hold out
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    enc = resolve_enc(a.enc)
    tok = AutoTokenizer.from_pretrained(enc)
    exs = [e for e in load_jsonl(f"{DATA}/{a.data}_train.jsonl", None) if len(e["gold"]) > 0]
    exs = exs[: len(exs) - a.holdout]
    lens = sorted({len(e["gold"]) for e in exs})
    cls = {n: i for i, n in enumerate(lens)}
    print(f"{len(exs)} training questions; lengths {dict(collections.Counter(len(e['gold']) for e in exs))}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(enc, num_labels=len(lens)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(exs) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def batch(qs):
        x = tok(qs, truncation=True, max_length=64, padding=True, return_tensors="pt")
        return {k: v.to(DEV) for k, v in x.items()}

    it = 0
    for ep in range(a.epochs):
        model.train()
        order = list(range(len(exs))); random.shuffle(order)
        for s in range(0, len(order), a.bs):
            b = [exs[i] for i in order[s: s + a.bs]]
            y = torch.tensor([cls[len(e["gold"])] for e in b], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**batch([e["question"] for e in b])).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            it += 1
            if it % 100 == 0:
                print(f"step {it}/{steps} loss {loss.item():.3f}", flush=True)
    model.eval()
    for sp in a.splits:
        ev = [e for e in load_jsonl(f"{DATA}/{sp}.jsonl", None) if len(e["gold"]) > 0]
        pred = {}
        with torch.no_grad():
            for s in range(0, len(ev), 64):
                b = ev[s: s + 64]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                    lg = model(**batch([e["question"] for e in b])).logits.float()
                for e, j in zip(b, lg.argmax(-1).tolist()):
                    pred[e["id"]] = lens[j]
        acc = collections.defaultdict(list)
        for e in ev:
            acc[len(e["gold"])].append(pred[e["id"]] == len(e["gold"]))
        tot = sum(sum(v) for v in acc.values()) / len(ev)
        print(f"LENCLF seed {a.seed} {sp}: accuracy {100 * tot:.1f} (n={len(ev)}); by length "
              + "  ".join(f"{h}: {100 * sum(v) / len(v):.1f}" for h, v in sorted(acc.items())), flush=True)
        json.dump(pred, open(f"{a.out}.{sp}.s{a.seed}.json", "w"))
    print("LENCLF_DONE", flush=True)


if __name__ == "__main__":
    main()
