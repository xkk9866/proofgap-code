"""RankGPT (Sun et al., EMNLP 2023) over each question's candidate pool.

Sliding-window listwise permutation, the published instruction, decoded by Qwen3-8B.
Window 10, stride 5. Passages the model does not name stay in their previous order.

python rankgpt.py --splits musique_dev,2wiki_dev,open_musique,open_2wiki --outdir $RES
"""
import argparse
import json
import os
import re

from vllm import LLM, SamplingParams

from data import load_jsonl
from evaluate import retrieval_metrics
from llm import DATA, MODEL, chat, passages_block

SYS = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
USER = """I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {question}.

{context}

Search Query: {question}.

Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only respond with the ranking results, do not say any word or explain."""


W, STRIDE = 10, 5  # RankGPT sliding window; a window of 10 fits the 2048-token context


def windows(n):
    if n <= W:
        return [0]
    starts = list(range(n - W, -1, -STRIDE))
    if starts[-1] != 0:
        starts.append(0)
    return starts


def rank_split(llm, tok, sp, exs, out, split):
    orders = [list(range(len(e["ctxs"]))) for e in exs]
    npass = max(len(windows(len(e["ctxs"]))) for e in exs)
    for step in range(npass):
        idx, prompts, wins = [], [], []
        for i, e in enumerate(exs):
            st = windows(len(e["ctxs"]))
            if step >= len(st):
                continue
            start = st[step]
            window = orders[i][start:start + W]
            prompts.append(chat(tok, SYS, USER.format(
                num=len(window), question=e["question"],
                context=passages_block(e["ctxs"], window, 60))))
            idx.append(i)
            wins.append((start, window))
        for c0 in range(0, len(prompts), 32):
            outs = llm.generate(prompts[c0:c0 + 32], sp)
            for o, i, (start, window) in zip(outs, idx[c0:c0 + 32], wins[c0:c0 + 32]):
                perm = permutation(o.outputs[0].text, len(window))
                orders[i][start:start + len(window)] = [window[j] for j in perm]
        print(split, "window", step + 1, "/", npass, flush=True)
    with open(out, "w") as f:
        for e, order in zip(exs, orders):
            f.write(json.dumps({"id": e["id"], "order": order}) + "\n")


def permutation(text, n):
    order = []
    for x in re.findall(r"\[(\d+)\]", text):
        i = int(x) - 1
        if 0 <= i < n and i not in order:
            order.append(i)
    order += [i for i in range(n) if i not in order]
    return order


def report(split, path):
    m = retrieval_metrics(split, path, by_hops=True)
    hops = m.pop("by_hops", {})
    print(f"EVAL {split} n={m['n']} FSR {m['FSR@2']:.1f}/{m['FSR@3']:.1f}/{m['FSR@4']:.1f}/{m['FSR@5']:.1f} "
          f"SetEM {m['SetEM@|G|']:.1f} SR4 {m['SR@4']:.1f}", flush=True)
    if split == "musique_dev":
        print("HOPS", {k: round(v, 1) for k, v in hops.items() if k.endswith("@2") or k.endswith("@3") or k.endswith("@4")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import torch
    free, total = torch.cuda.mem_get_info()
    # Fraction of the whole card. Leave 0.8 GiB so the ranker can use the memory that is actually free.
    util = max(0.50, min(0.96, (total - int(0.8 * 2 ** 30)) / total))
    print(f"cuda free {free/2**30:.1f} GiB, util {util:.2f}", flush=True)
    llm = LLM(MODEL, quantization="awq_marlin", max_model_len=2048, seed=0,
              gpu_memory_utilization=util, enforce_eager=True, max_num_seqs=8)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    for split in a.splits.split(","):
        out = f"{a.outdir}/{split}.rankgpt.jsonl"
        exs = load_jsonl(f"{DATA}/{split}.jsonl", a.limit or None)
        if os.path.exists(out) and not a.limit and sum(1 for _ in open(out)) >= len(exs):
            print("skip", split, flush=True)
            report(split, out)
            continue
        rank_split(llm, tok, sp, exs, out, split)
        if not a.limit:
            report(split, out)


if __name__ == "__main__":
    main()
