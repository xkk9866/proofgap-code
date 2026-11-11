"""Qwen3-Reranker-4B (Qwen, 2025) over each question's candidate pool.

The published cross-encoder: default instruction, raw logits, higher is more relevant.
No task-specific instruction is added.

python qwen3rr.py --splits musique_dev,2wiki_dev,open_musique,open_2wiki --outdir $RES
"""
import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from data import ctx_str, load_jsonl
from evaluate import retrieval_metrics

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
MODEL = "/data/yaokang/models/Qwen3-Reranker-4B"


def report(split, path):
    m = retrieval_metrics(split, path, by_hops=True)
    hops = m.pop("by_hops", {})
    print(f"EVAL {split} n={m['n']} FSR {m['FSR@2']:.1f}/{m['FSR@3']:.1f}/{m['FSR@4']:.1f}/{m['FSR@5']:.1f} "
          f"SetEM {m['SetEM@|G|']:.1f} SR4 {m['SR@4']:.1f}", flush=True)
    if split == "musique_dev":
        print("HOPS", {k: round(v, 1) for k, v in hops.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rr = CrossEncoder(MODEL, device="cuda", max_length=1024, automodel_args={"torch_dtype": torch.float16})
    for split in a.splits.split(","):
        out = f"{a.outdir}/{split}.qwen3rr.jsonl"
        exs = load_jsonl(f"{DATA}/{split}.jsonl", a.limit or None)
        if os.path.exists(out) and not a.limit and sum(1 for _ in open(out)) >= len(exs):
            print("skip", split, flush=True)
            report(split, out)
            continue
        done = {}  # questions already ranked by an interrupted run of this split
        if os.path.exists(out):
            for line in open(out):
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except Exception:
                    pass
        with open(out, "w") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(split, "resuming after", len(done), flush=True)
            for i, e in enumerate(exs):
                if e["id"] in done:
                    continue
                docs = [ctx_str(c) for c in e["ctxs"]]
                scores = np.asarray(rr.predict([(e["question"], d) for d in docs], batch_size=int(os.environ.get("RR_BS", 16)), show_progress_bar=False))
                order = [int(j) for j in np.argsort(-scores)]
                f.write(json.dumps({"id": e["id"], "order": order}) + "\n")
                if i % 10 == 0:
                    f.flush()
                if i % 100 == 0:
                    print(split, i, "/", len(exs), flush=True)
        if not a.limit:
            report(split, out)


if __name__ == "__main__":
    main()
