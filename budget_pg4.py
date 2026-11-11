"""Training-budget study: a 4-epoch run whose epoch is chosen on the SOURCE hold-out only (no dev label is used).

python budget_pg4.py <pg|beam> <ckpt_dir_with_ep1..ep3_and_final> <results_dir>

Writes <results>/budget_<method>.json (hold-out Set-EM of every epoch and the chosen one) and the dev / open-corpus
selections of the chosen epoch as <results>/musique_dev.<method>_4ep.jsonl and <results>/open_musique.<method>_4ep.jsonl.
"""
import json
import os
import subprocess
import sys

from evaluate import retrieval_metrics

PY = sys.executable


def infer(method, ckpt, split, out):
    if os.path.exists(out):
        return
    flags = ["--pg4", "2", "--bs", "1", "--cand", "member", "--front", "pooled"] if method == "pg" else ["--search", "2", "--no_accum", "--bs", "1"]
    subprocess.run([PY, "infer.py", "--ckpt", ckpt, "--split", split, "--out", out] + flags, check=True)


def main(method, ck, R):
    epochs = {f"ep{i}": f"{ck}/ep{i}" for i in (1, 2, 3) if os.path.exists(f"{ck}/ep{i}/model.pt")}
    epochs["ep4"] = ck
    hold = {}
    for name, path in epochs.items():
        out = f"{R}/musique_hold.{method}_4ep_{name}.jsonl"
        infer(method, path, "musique_hold", out)
        hold[name] = retrieval_metrics("musique_hold", out)["SetEM@|G|"]
    best = max(hold, key=hold.get)
    for split in ("musique_dev", "open_musique"):
        infer(method, epochs[best], split, f"{R}/{split}.{method}_4ep.jsonl")
    json.dump({"method": method, "hold_SetEM": hold, "chosen": best}, open(f"{R}/budget_{method}.json", "w"), indent=1)
    print(method, hold, "chosen", best)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
