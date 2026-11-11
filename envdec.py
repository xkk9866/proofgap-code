"""Monotone-envelope decision: P0*(S) = max_{T subset of S, T read} P0(T)  (r(S) <= r(T) for T subset of S)."""
import json, sys
from evaluate import gold_map, load
R = "/data/yaokang/ercrag/results"
def envelope(cands):
    seen = {}
    sets, ver = {}, {}
    for k in sorted(cands, key=int):
        for S, lp, ng in cands[k]:
            seen[frozenset(S)] = max(lp, seen.get(frozenset(S), -1e9))
        best = None
        for S, lp, ng in cands[k]:
            fs = frozenset(S)
            env = max(v for T, v in seen.items() if T <= fs)
            key = (env, lp, ng)
            if best is None or key > best[0]:
                best = (key, S)
        sets[k], ver[k] = sorted(best[1]), best[0][0]
    return sets, ver
if __name__ == "__main__":
    sp, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(f"{R}/{sp}.{out}.jsonl", "w") as f:
        for r in load(f"{R}/{sp}.{key}.jsonl"):
            r["sets"], r["verdict"] = envelope(r["cands"])
            f.write(json.dumps(r) + "\n")
