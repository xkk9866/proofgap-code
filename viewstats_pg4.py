"""Which view decides?  Statistics of the member views stored by focus_search (cands[*][5]) on 2-hop questions whose gold
pair is among the depth-2 candidates: rung-0 logits of the gold pair's views, and in near-proofs {g, x} the view through
the impostor x versus the view through the annotated member g (CPU only).

RES=/data/yaokang/ercrag/results_pg4 python viewstats_pg4.py musique_dev open_musique   -> $RES/viewstats.json
"""
import json
import os
import statistics as st
import sys

from evaluate import gold_map, load

R = os.environ.get("RES", "/data/yaokang/ercrag/results_pg4")


def stats(split, key="pg"):
    G = gold_map(split)
    gold_v, near_x, near_g, lower, n = [], [], [], 0, 0
    for r in load(f"{R}/{split}.{key}.jsonl"):
        e = G[r["id"]]; g = set(e["gold"])
        if len(g) != 2 or "2" not in r["cands"]:
            continue
        L = r["cands"]["2"]
        gold = [x for x in L if set(x[0]) == g and len(x) >= 6]
        if not gold:
            continue
        n += 1
        gold_v += list(gold[0][5].values())
        for x in L:
            S = set(x[0])
            if len(S & g) == 1 and len(x) >= 6:
                vx, vg = x[5].get(str(next(iter(S - g)))), x[5].get(str(next(iter(S & g))))
                if vx is not None and vg is not None:
                    near_x.append(vx); near_g.append(vg); lower += vx < vg
    return {"n_questions": n, "n_near": len(near_x),
            "gold_mean_u0": st.mean(gold_v), "gold_median_u0": st.median(gold_v),
            "near_impostor_mean_u0": st.mean(near_x), "near_impostor_median_u0": st.median(near_x),
            "near_gold_mean_u0": st.mean(near_g), "near_gold_median_u0": st.median(near_g),
            "impostor_view_lower_pct": 100 * lower / max(1, len(near_x))}


def main(splits):
    out = {sp: stats(sp) for sp in splits}
    json.dump(out, open(f"{R}/viewstats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["musique_dev", "open_musique"])
