"""Evaluate full-dev result files restricted to the ids of a subset split.  python subeval.py musique_dev_s600 musique_dev key1 key2 ..."""
import json, sys, tempfile
from evaluate import retrieval_metrics
R = "/data/yaokang/ercrag/results"
sub, full = sys.argv[1], sys.argv[2]
ids = {json.loads(l)["id"] for l in open(f"/data/yaokang/ercrag/data/proc/{sub}.jsonl")}
for k in sys.argv[3:]:
    p = f"{R}/{full}.{k}.jsonl" if "/" not in k else k
    rows = [l for l in open(p) if json.loads(l)["id"] in ids]
    t = tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl"); t.writelines(rows); t.close()
    m = retrieval_metrics(sub, t.name)
    print(f"{k:22s}", " ".join(f"{m[x]:.1f}" for x in ("FSR@2", "FSR@3", "FSR@4", "FSR@5", "SetEM@|G|")), m["n"])
