"""INT4 (AWQ) Qwen3-8B LLM components with vLLM: QA reader over selected evidence, and a SetR-style
zero-shot LLM set selector (information-requirement identification + set selection).

python llm.py read   --split musique_dev --sel results/x.jsonl --key order --k 4 --out results/qa_x.jsonl
python llm.py select --split musique_dev --out results/musique_dev.llmset.jsonl
"""
import argparse
import json
import os
import re

from vllm import LLM, SamplingParams

from data import load_jsonl

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
MODEL = os.environ.get("LLM_MODEL", "/home/yaokang/models/Qwen3-8B-AWQ")  # INT4 AWQ weights

READ_SYS = ("You are a precise question answering assistant. Use the given passages to answer the multi-hop question. "
            "Reason briefly if needed, then finish with a line 'Answer: <short answer>' where the short answer is an entity, "
            "date, number or a few words, with no explanation.")
# SetR official selection_IRI prompt (github.com/LGAI-Research/SetR, generate_data.py)
SEL_SYS = "You are RankLLM, an intelligent assistant that can select passages based on their relevancy to the query."
SEL_USER = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""


def llm_util(default):
    """vLLM's utilisation is a share of the WHOLE card, other processes included. LLM_UTIL=auto takes what is
    in use now plus 11.5 GB for the INT4 8B reader and its KV cache."""
    v = os.environ.get("LLM_UTIL", default)
    if v != "auto":
        return float(v)
    import torch
    free, total = torch.cuda.mem_get_info()
    return min(0.92, (total - free + 11.5 * 2 ** 30) / total)


def passages_block(ctxs, idxs, words=120):
    out = []
    for r, i in enumerate(idxs):
        c = ctxs[i]
        t = " ".join(c["text"].split()[:words])
        out.append(f"[{r + 1}] {c['title']}: {t}")
    return "\n".join(out)


def chat(tok, sys, user):
    return tok.apply_chat_template([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def parse_answer(t):
    m = re.findall(r"Answer:\s*(.*)", t)
    a = (m[-1] if m else t.strip().split("\n")[-1]).strip()
    return a.strip(" .\"'*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--split", required=True)
    ap.add_argument("--sel", default=None)
    ap.add_argument("--key", default="order")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--jobs", default=None, help="json list of [sel, key, k, out] to batch several reads in one LLM load")
    a = ap.parse_args()
    exs = load_jsonl(f"{DATA}/{a.split}.jsonl", None if os.environ.get("QA_RANDOM") else a.limit)
    if os.environ.get("QA_RANDOM"):  # uniform random sample (seed 0): keeps the split's hop distribution
        import random
        exs = random.Random(0).sample(exs, min(a.limit, len(exs)))
    llm = LLM(MODEL, quantization="awq_marlin", max_model_len=8192, seed=0,
              gpu_memory_utilization=llm_util("0.45"))
    tok = llm.get_tokenizer()
    if a.mode == "select":
        sp = SamplingParams(temperature=0.0, max_tokens=700)
        topn = int(os.environ.get("SEL_TOPN", "0"))  # SetR reads the head of the first-stage ranking in one prompt
        if topn:
            exs = [dict(e, ctxs=e["ctxs"][:topn]) for e in exs]
        prompts = [chat(tok, SEL_SYS, SEL_USER.format(num=len(e["ctxs"]), question=e["question"],
                                                      context=passages_block(e["ctxs"], range(len(e["ctxs"])), 90)))
                   for e in exs]
        outs = llm.generate(prompts, sp)
        with open(a.out, "w") as f:
            for e, o in zip(exs, outs):
                t = o.outputs[0].text
                m = t.split("Final Selection:")
                sel = []
                if len(m) > 1:
                    for x in re.findall(r"\[(\d+)\]", m[-1]):
                        i = int(x) - 1
                        if 0 <= i < len(e["ctxs"]) and i not in sel:
                            sel.append(i)
                f.write(json.dumps({"id": e["id"], "order": sel, "raw": t}) + "\n")
        return
    jobs = json.loads(open(a.jobs).read()) if a.jobs else [[a.sel, a.key, a.k, a.out]]
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    for sel_path, key, k, out_path in jobs:
        sel = {r["id"]: r for r in load_jsonl(sel_path)}
        prompts = []
        for e in exs:
            r = sel.get(e["id"], {})
            if key.startswith("set"):  # per-depth answer of a set-level search, e.g. "set4"
                o = r.get("sets", {}).get(key[3:], r.get("order", [])[: int(key[3:])])
            else:
                o = r.get(key, [])
                o = o[:k] if k else o
            prompts.append(chat(tok, READ_SYS, f"Passages:\n{passages_block(e['ctxs'], o, 400)}\n\nQuestion: {e['question']}"))
        outs = llm.generate(prompts, sp)
        with open(out_path, "w") as f:
            for e, o in zip(exs, outs):
                f.write(json.dumps({"id": e["id"], "pred": parse_answer(o.outputs[0].text), "raw": o.outputs[0].text}) + "\n")
        print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
