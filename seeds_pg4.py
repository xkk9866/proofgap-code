"""Training-seed statistics for the v4 MuSiQue models (CPU only).

python seeds_pg4.py <results_pg4> <results_pg>   -> prints JSON {split: {method: {metric: {mean, sd, values}}, diff: ...}}

Seed 0 of ProofGap is the reference run (<results_pg4>/musique_dev.pg.jsonl); seeds 1 and 2 are under s1/ and s2/.
Seed 0 of Beam Retrieval is the reference run of the shared protocol (<results_pg>/musique_dev.beam.jsonl); its seeds
1 and 2 were trained with the same recipe and copied to s1/ and s2/.  SD uses ddof=1 (sample standard deviation over
training runs); `diff` is the matched-seed ProofGap minus Beam difference.
"""
import json
import os
import sys

import numpy as np

from evaluate import retrieval_metrics

MET = ["FSR@2", "FSR@3", "FSR@4", "FSR@5", "SetEM@|G|"]


def main(R, R3):
    out = {}
    for split in ("musique_dev", "open_musique"):
        paths = {"pg": [f"{R}/{split}.pg.jsonl", f"{R}/s1/{split}.pg.jsonl", f"{R}/s2/{split}.pg.jsonl"],
                 "beam": [f"{R3}/{split}.beam.jsonl", f"{R}/s1/{split}.beam.jsonl", f"{R}/s2/{split}.beam.jsonl"]}
        res = {}
        for m, ps in paths.items():
            ps = [p for p in ps if os.path.exists(p)]
            vals = [retrieval_metrics(split, p) for p in ps]
            res[m] = {k: {"values": [v[k] for v in vals], "mean": float(np.mean([v[k] for v in vals])),
                          "sd": float(np.std([v[k] for v in vals], ddof=1)) if len(vals) > 1 else float("nan"), "n_seeds": len(vals)} for k in MET}
        if res["pg"][MET[0]]["n_seeds"] == res["beam"][MET[0]]["n_seeds"]:
            res["diff"] = {}
            for k in MET:
                d = np.array(res["pg"][k]["values"]) - np.array(res["beam"][k]["values"])
                res["diff"][k] = {"values": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"), "min": float(d.min())}
        out[split] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
