import json, sys, collections, math
from evaluate import gold_map, load
sp, key = sys.argv[1], sys.argv[2]
G = gold_map(sp); c = collections.Counter(); pc=[];pi=[]
for r in load(f"/data/yaokang/ercrag/results/{sp}.{key}.jsonl"):
    g = set(G[r["id"]]["gold"])
    for k in ("3","4"):
        cs = r["cands"][k]
        comp = [x for x in cs if g <= set(x[0])]
        if not comp: continue
        ch = max(cs, key=lambda x: x[1])
        if g <= set(ch[0]): c[k,"ok"]+=1
        else:
            c[k, "err_gap%d" % len(g-set(ch[0])), "hops%d"%len(g)]+=1
            pc.append(max(x[1] for x in comp)); pi.append(ch[1])
for k,v in sorted(c.items()): print(k,v)
import numpy as np
print("mean P0 best complete", np.mean(np.exp(pc)), "chosen incomplete", np.mean(np.exp(pi)))
