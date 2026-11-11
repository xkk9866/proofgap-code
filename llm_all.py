"""Run the LLM stage for every split in one vLLM load per split:
SetR-style selection, then QA reading for each available method at budget k=4 (and native SetR sets).

Env: QA_TAG (suffix of qa output files, e.g. "_q25" for a second reader), LLM_MODEL (reader weights),
QA_METHODS (comma list restricting the methods), NO_SELECT=1 (skip SetR selection).
"""
import json
import os
import subprocess
import sys

R = __import__("os").environ.get("RES", "/data/yaokang/ercrag/results")
PY = sys.executable
K = 4
TAG = os.environ.get("QA_TAG", "")
METHODS = [("rerank", "order", K), ("ce", "set4", 0), ("beam_w1", "set4", 0), ("beam_w2", "set4", 0),
           ("beam_off", "set4", 0), ("prover", "set4", 0), ("prover_adapt", "order", 0),
           ("deficit", "set4", 0), ("deficit_adapt", "order", 0), ("ladder", "set4", 0), ("ladder_adapt", "order", 0),
           ("llmset", "order", 0), ("gap", "set4", 0), ("gap_adapt", "order", 0), ("beam_offL", "set4", 0), ("ladder_cal", "order", 0), ("beam_off_s1", "set4", 0), ("ircot", "order", K),
           # ProofGap v3 pipeline (RES=results_pg): new supervised models and the adaptive operating point
           ("pg", "set4", 0), ("pg_adapt", "order", 0), ("beam", "set4", 0)]
if os.environ.get("QA_METHODS"):
    keep = os.environ["QA_METHODS"].split(",")
    METHODS = [m for m in METHODS if m[0] in keep]

from analysis import mkadapt, mkadapt_cal

for split in sys.argv[1:]:
    if os.environ.get("NO_ADAPT"):  # v3 pipeline: adaptive sets come from stopping.py --write_adapt
        pass
    elif os.path.exists(f"{R}/{split}.prover.jsonl"):
        mkadapt(split, "prover", 0.5)
    for m in ("deficit", "ladder", "gap"):
        if not os.environ.get("NO_ADAPT") and os.path.exists(f"{R}/{split}.{m}.jsonl"):
            mkadapt(split, m, 0.5)
    if not os.environ.get("NO_ADAPT") and os.path.exists(f"{R}/{split}.ladder.jsonl"):
        mkadapt_cal(split, "ladder", 0.5)
    if not os.environ.get("NO_SELECT") and not os.path.exists(f"{R}/{split}.llmset.jsonl"):
        subprocess.run([PY, "llm.py", "select", "--split", split, "--out", f"{R}/{split}.llmset.jsonl"], check=True)
    jobs = []
    for m, key, k in METHODS:
        if os.path.exists(f"{R}/{split}.{m}.jsonl") and not os.path.exists(f"{R}/qa_{split}.{m}{TAG}.jsonl"):
            jobs.append([f"{R}/{split}.{m}.jsonl", key, k, f"{R}/qa_{split}.{m}{TAG}.jsonl"])
    gold = f"{R}/{split}.gold.jsonl"
    with open(gold, "w") as f:
        for l in open(f"/data/yaokang/ercrag/data/proc/{split}.jsonl"):
            e = json.loads(l)
            f.write(json.dumps({"id": e["id"], "order": [g for g in e["gold"] if g < 1000]}) + "\n")
    if not os.path.exists(f"{R}/qa_{split}.gold{TAG}.jsonl"):
        jobs.append([gold, "order", 0, f"{R}/qa_{split}.gold{TAG}.jsonl"])
    if os.environ.get("QA_CLOSED") and not os.path.exists(f"{R}/qa_{split}.closed{TAG}.jsonl"):
        jobs.append([gold, "none", 0, f"{R}/qa_{split}.closed{TAG}.jsonl"])
    if jobs:
        jf = f"{R}/llmjobs_{split}{TAG}.json"
        json.dump(jobs, open(jf, "w"))
        subprocess.run([PY, "llm.py", "read", "--split", split, "--out", "x", "--jobs", jf], check=True)
