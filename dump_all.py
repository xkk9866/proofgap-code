"""Dump every metric the paper needs into results/all_metrics.json (no GPU)."""
import glob
import json
import os

from analysis import adaptive, boot, decomp
from gapstats import descent
from evaluate import calibration, qa_metrics, retrieval_metrics

R = "/data/yaokang/ercrag/results"
SPLITS = ["musique_dev", "hotpotqa_dev", "2wiki_dev", "open_musique", "open_hotpotqa", "open_2wiki"]


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # keep going, report the failure
        return {"error": repr(e)}


out = {"ret": {}, "qa": {}, "hops": {}, "decomp": {}, "adapt": {}, "boot": {}, "cal": {}}
for sp in SPLITS:
    for p in sorted(glob.glob(f"{R}/{sp}.*.jsonl")):
        k = p[len(f"{R}/{sp}."):-6]
        if k in ("gold",):
            continue
        out["ret"][f"{sp}|{k}"] = safe(retrieval_metrics, sp, p)
    for p in sorted(glob.glob(f"{R}/qa_{sp}.*.jsonl")):
        k = p[len(f"{R}/qa_{sp}."):-6]
        out["qa"][f"{sp}|{k}"] = safe(qa_metrics, sp, p)
    for k in ("ladder", "deficit", "prover", "ladder_vonly", "deficit_vonly", "prover_vonly"):
        if os.path.exists(f"{R}/{sp}.{k}.jsonl"):
            out["decomp"][f"{sp}|{k}"] = safe(decomp, sp, k)
    for k in ("ladder", "deficit", "prover"):
        if os.path.exists(f"{R}/{sp}.{k}.jsonl"):
            out["adapt"][f"{sp}|{k}"] = safe(adaptive, sp, k)
            out["cal"][f"{sp}|{k}"] = {kk: safe(calibration, sp, f"{R}/{sp}.{k}.jsonl", kk) for kk in (2, 3, 4)}
    for b in ("beam_off", "beam_w1", "beam_w2", "prover", "rerank", "ce", "bm25", "dense", "mmr", "llmsetfill", "ircot"):
        for a in ("ladder", "deficit", "prover"):
            if a != b and os.path.exists(f"{R}/{sp}.{a}.jsonl") and os.path.exists(f"{R}/{sp}.{b}.jsonl"):
                out["boot"][f"{sp}|{a}|{b}"] = safe(boot, sp, a, b)
for k in ("deficit_1ep", "prover_1ep", "beam_1ep", "deficit_1ep_vonly"):
    if os.path.exists(f"{R}/musique_dev.{k}.jsonl"):
        out["decomp"][f"musique_dev|{k}"] = safe(decomp, "musique_dev", k)
for k in ("rerank", "ce", "beam_off", "beam_w1", "beam_w2", "erc2_w1", "prover", "deficit", "ladder"):
    p = f"{R}/musique_dev.{k}.jsonl"
    if os.path.exists(p):
        out["hops"][k] = safe(retrieval_metrics, "musique_dev", p, ks=(2, 3, 4), by_hops=True)
for b in ("beam_1ep", "prover_1ep"):
    out["boot"][f"musique_dev|deficit_1ep|{b}"] = safe(boot, "musique_dev", "deficit_1ep", b)
out["gapdesc"] = {}
for k in ("gap1_ordinal", "gap1_ordinal_s1", "gap1_binary", "gap1_binary_s1", "gap1_multiclass", "gap1d_ordinal", "gap1d_binary"):
    if os.path.exists(f"{R}/musique_dev.{k}.jsonl"):
        out["gapdesc"][k] = safe(descent, "musique_dev", k)
for f in ("latency2", "calib", "calib_ladder", "audit_musique", "audit_hotpotqa", "calstop_ladder",
          "audit2_musique_dev", "audit2_hotpotqa_dev", "audit2_2wiki_dev"):
    try:
        out[f] = json.load(open(f"{R}/{f}.json"))
    except Exception as e:
        out[f] = {"error": repr(e)}
json.dump(out, open(f"{R}/all_metrics.json", "w"), indent=1, default=float)
print("ok", {k: len(v) for k, v in out.items() if isinstance(v, dict)})
