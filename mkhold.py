import json
D = "/data/yaokang/ercrag/data/proc"
for ds, lo in (("musique", None), ("hotpotqa", 20000), ("2wiki", 20000)):
    ex = [json.loads(l) for l in open(f"{D}/{ds}_train.jsonl")]
    ex = [e for e in ex if e["gold"]]
    hold = ex[-1000:] if lo is None else ex[lo:lo + 1000]
    with open(f"{D}/{ds}_hold.jsonl", "w") as f:
        for e in hold:
            f.write(json.dumps(e) + "\n")
    print(ds, len(ex), len(hold))
