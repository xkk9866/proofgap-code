"""IRCoT baseline (Trivedi et al., ACL 2023), reproduced over each question's candidate pool.

Interleaves retrieval and chain-of-thought: retrieve K passages for the question; then repeatedly let the LLM write
the next CoT sentence given the passages collected so far, and use that sentence as the next retrieval query, adding
its top-K new passages; stop when the CoT states the answer or after T steps. As in the paper, the evidence set is
the passages collected, in collection order. The retriever is bge-reranker-v2-m3 (stronger than the original BM25),
the reasoner the same INT4 Qwen3-8B as SetR. Fixed budgets fill the remainder with the question's reranker order.

python ircot.py --split musique_dev --out results/musique_dev.ircot.jsonl
"""
import argparse
import json
import re

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from vllm import LLM, SamplingParams

from data import ctx_str, load_jsonl
from llm import DATA, MODEL, chat, llm_util, passages_block

K, T = 2, 4  # passages per retrieval, reasoning steps (IRCoT defaults: K in {2..15}, stop on "answer is")
SYS = ("You answer multi-hop questions by reasoning step by step over the given passages. Each time you are asked, "
       "write ONLY the next single sentence of the reasoning chain. When the chain reaches the answer, write the "
       "sentence 'So the answer is: <short answer>.'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os
    if os.path.exists(a.out):  # resumable: another worker already produced this split
        return
    exs = load_jsonl(f"{DATA}/{a.split}.jsonl")
    llm = LLM(MODEL, quantization="awq_marlin", max_model_len=8192, seed=0, gpu_memory_utilization=llm_util("0.6"))
    tok = llm.get_tokenizer()
    rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, automodel_args={"torch_dtype": torch.float16})
    docs = [[ctx_str(c) for c in e["ctxs"]] for e in exs]

    def retrieve(qs, idx):  # reranker scores of every candidate for each (question index, query)
        pairs = [(q, d) for q, i in zip(qs, idx) for d in docs[i]]
        s = rr.predict(pairs, batch_size=128, show_progress_bar=False)
        out, p = [], 0
        for i in idx:
            out.append(np.asarray(s[p:p + len(docs[i])])); p += len(docs[i])
        return out

    qscore = retrieve([e["question"] for e in exs], range(len(exs)))
    got = [[int(j) for j in np.argsort(-s)[:K]] for s in qscore]
    cot = [[] for _ in exs]
    live = list(range(len(exs)))
    sp = SamplingParams(temperature=0.0, max_tokens=96, stop=["\n"])
    for t in range(T):
        prompts = [chat(tok, SYS, f"Passages:\n{passages_block(exs[i]['ctxs'], got[i], 200)}\n\nQuestion: {exs[i]['question']}\n"
                                  f"Reasoning so far: {' '.join(cot[i]) or '(none)'}\nNext sentence:") for i in live]
        outs = llm.generate(prompts, sp)
        nxt, qs = [], []
        for i, o in zip(live, outs):
            s = o.outputs[0].text.strip()
            s = re.split(r"(?<=[.!?])\s", s)[0] if s else s
            cot[i].append(s)
            if s and "answer is" not in s.lower():
                nxt.append(i); qs.append(s)
        if t == T - 1 or not nxt:
            break
        for i, s in zip(nxt, retrieve(qs, nxt)):
            got[i] += [int(j) for j in np.argsort(-s) if int(j) not in got[i]][:K]
        live = nxt
        print("step", t, "live", len(live), flush=True)
    with open(a.out, "w") as f:
        for e, g, s, c in zip(exs, got, qscore, cot):
            o = g + [int(j) for j in np.argsort(-s) if int(j) not in g]
            f.write(json.dumps({"id": e["id"], "order": o, "native": len(g), "cot": c}) + "\n")


if __name__ == "__main__":
    main()
