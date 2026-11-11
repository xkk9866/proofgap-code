"""Compare result files on the questions they share: FSR@2..5, Set-EM, coverage kappa_k (if cands), by type."""
import collections, json, sys
D = "/data/yaokang/ercrag/data/proc"
def load(p): return [json.loads(l) for l in open(p)]
split = sys.argv[1]; files = sys.argv[2:]
G = {e["id"]: e for e in load(f"{D}/{split}.jsonl")}
rows = [{r["id"]: r for r in load(f)} for f in files]
ids = [q for q in G if all(q in R for R in rows) and G[q]["gold"]]
print(split, "n =", len(ids))
for f, R in zip(files, rows):
    m = collections.defaultdict(list)
    for q in ids:
        g = set(G[q]["gold"]); r = R[q]
        pick = (lambda k: r["sets"].get(str(k), [])) if "sets" in r else (lambda k: r["order"][:k])
        for k in (2, 3, 4, 5):
            m[f"@{k}"].append(g <= set(pick(k)))
            if "cands" in r and str(k) in r["cands"]:
                m[f"k{k}"].append(any(g <= set(x[0]) for x in r["cands"][str(k)]))
        m["SetEM"].append(g <= set(pick(len(g))))
        t = G[q].get("type", "")[:6]
        m[f"SetEM[{t}]"].append(g <= set(pick(len(g))))
    print(f"  {f.split('/')[-1]:40s}", " ".join(f"{k}={100*sum(v)/len(v):.1f}" for k, v in m.items() if not k.startswith("SetEM[")),
          "|", " ".join(f"{k[6:-1]}={100*sum(v)/len(v):.1f}" for k, v in sorted(m.items()) if k.startswith("SetEM[")))
