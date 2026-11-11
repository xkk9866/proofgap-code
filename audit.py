"""Label audit: do 'incomplete' near-proofs secretly suffice?

For each question we take, from the retriever's own lattice (AUDIT_KEY, default ladder) for a uniform random sample of questions at k=|G|, the highest-verdict INCOMPLETE candidate
(the near-proof the verdict had to reject) and the gold set. The INT4 reader answers from each.
If a near-proof yields the gold answer about as often as the gold set, annotation-based labels would be
unreliable; if it is far lower, near-proof negatives are genuinely insufficient evidence.

python audit.py --split musique_dev --n 1000 --out results/audit_musique.json
"""
import argparse
import json
import os
import random

from vllm import LLM, SamplingParams

from evaluate import f1, norm
from llm import MODEL, READ_SYS, chat, parse_answer, passages_block

JUDGE_SYS = ("You check evidence. Decide whether the passages, without outside knowledge, state facts from which the "
             "reference answer to the question follows. Reply with one word: yes or no.")
D = "/data/yaokang/ercrag/data/proc"
R = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")


def llm_util(default):
    """vLLM's utilisation is a share of the WHOLE card, other processes included. LLM_UTIL=auto takes what is
    in use now plus 11.5 GB for the INT4 8B reader and its KV cache."""
    v = os.environ.get("LLM_UTIL", default)
    if v != "auto":
        return float(v)
    import torch
    free, total = torch.cuda.mem_get_info()
    return min(0.92, (total - free + 11.5 * 2 ** 30) / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="musique_dev")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ex = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/{a.split}.jsonl")}
    rows = [json.loads(l) for l in open(f"{R}/{a.split}.{os.environ.get('AUDIT_KEY', 'ladder')}.jsonl")]
    random.Random(0).shuffle(rows)  # uniform random sample of questions (all hop counts)
    sets = []
    for r in rows:
        e = ex[r["id"]]
        k = str(len(e["gold"]))
        if k not in r.get("cands", {}):
            continue
        g = set(e["gold"])
        inc = [c for c in r["cands"][k] if not g <= set(c[0])]
        if not inc:
            continue
        near = max(inc, key=lambda c: c[1])[0]  # the incomplete set the decision found most convincing
        sets.append((e, sorted(g), near))
        if len(sets) >= a.n:
            break
    llm = LLM(MODEL, quantization="awq_marlin", max_model_len=8192, seed=0,
              gpu_memory_utilization=llm_util("0.8"))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    conds = ("gold", "near", "closed")
    prompts = []
    for e, g, nr in sets:
        for S in (g, nr):
            prompts.append(chat(tok, READ_SYS, f"Passages:\n{passages_block(e['ctxs'], S, 400)}\n\nQuestion: {e['question']}"))
        prompts.append(chat(tok, READ_SYS, f"Passages:\n(none; answer from your own knowledge)\n\nQuestion: {e['question']}"))
    # independent sufficiency check: given the reference answer, do the passages alone establish it?
    for e, g, nr in sets:
        for S in (g, nr):
            prompts.append(chat(tok, JUDGE_SYS, f"Passages:\n{passages_block(e['ctxs'], S, 400)}\n\nQuestion: {e['question']}\n"
                                                f"Reference answer: {e['answers'][0]}"))
    outs = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    n = len(sets)
    res = {c: [] for c in conds}
    for i, (e, g, nr) in enumerate(sets):
        for j, c in enumerate(conds):
            p = parse_answer(outs[3 * i + j])
            res[c].append(max(float(norm(p) == norm(x)) for x in e["answers"]))
    suff = {"gold": [], "near": []}
    for i in range(n):
        for j, c in enumerate(("gold", "near")):
            suff[c].append(float("yes" in outs[3 * n + 2 * i + j].strip().lower()[-40:]))
    summ = {f"EM_{c}": 100 * sum(v) / n for c, v in res.items()}
    summ["near_only_correct"] = 100 * sum(1 for x, y in zip(res["gold"], res["near"]) if y and not x) / n
    summ["near_correct_closed_wrong"] = 100 * sum(1 for y, z in zip(res["near"], res["closed"]) if y and not z) / n
    summ["suff_gold"] = 100 * sum(suff["gold"]) / n
    summ["suff_near"] = 100 * sum(suff["near"]) / n
    summ["suff_near_and_correct_closed_wrong"] = 100 * sum(1 for s_, y, z in zip(suff["near"], res["near"], res["closed"]) if s_ and y and not z) / n
    summ["n"] = n
    print(summ)
    json.dump(summ, open(a.out, "w"))

if __name__ == "__main__":
    main()
