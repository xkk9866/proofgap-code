"""Model selection on held-out TRAINING questions (never on dev).

python selbest.py musique_hold gap_ep1 gap_ep2 gap_ep3   -> prints the key with the best mean of FSR@2..5 and Set-EM
"""
import sys

import numpy as np

from evaluate import retrieval_metrics

R = "/data/yaokang/ercrag/results"
split, keys = sys.argv[1], sys.argv[2:]
score = {}
for k in keys:
    m = retrieval_metrics(split, f"{R}/{split}.{k}.jsonl")
    score[k] = float(np.mean([m[x] for x in ("FSR@2", "FSR@3", "FSR@4", "FSR@5", "SetEM@|G|")]))
    print(k, round(score[k], 2), file=sys.stderr)
print(max(keys, key=lambda k: score[k]))
