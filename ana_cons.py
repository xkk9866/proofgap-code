"""How should the member views of one set be combined?  python ana_cons.py SPLIT PG.jsonl

Set-EM (budget = proof length) when the depth-|G| candidates are ranked by different consensus rules over the rung-0
logits of their member views: mean logit (the trained consensus), min, product of probabilities, max, median; and by the
expected deficit of the consensus ladder.
"""
import math
import sys
from collections import defaultdict

from evaluate import gold_map, load


def ls(x):
    return -math.log1p(math.exp(-x)) if x > -30 else x


def main():
    split, path = sys.argv[1:3]
    G = gold_map(split)
    rules = {
        "mean": lambda x: x[3][0],
        "min": lambda x: min(x[5].values()),
        "prod": lambda x: sum(ls(v) for v in x[5].values()),
        "gmean": lambda x: sum(ls(v) for v in x[5].values()) / len(x[5]),
        "max": lambda x: max(x[5].values()),
        "median": lambda x: sorted(x[5].values())[len(x[5]) // 2],
        "E[r]": lambda x: -sum(1 / (1 + math.exp(u)) for u in x[3]),
        "mean+min": lambda x: x[3][0] + min(x[5].values()),
    }
    em = defaultdict(list); hop = defaultdict(list)
    n = 0
    for r in load(path):
        e = G.get(r["id"])
        if not e or not e["gold"]:
            continue
        g = frozenset(e["gold"]); h = len(g); n += 1
        L = r["cands"].get(str(h), [])
        for name, f in rules.items():
            S = frozenset(max(L, key=lambda x: (f(x), x[2]))[0]) if L else frozenset()
            em[name].append(float(S == g)); hop[(name, h)].append(float(S == g))
    print(f"{split} n={n}")
    for name in rules:
        print(f"  {name:9s} Set-EM {100 * sum(em[name]) / n:.1f}  by hops "
              + " ".join(f"{h}:{100 * sum(v) / len(v):.1f}" for (nn, h), v in sorted(hop.items()) if nn == name))


if __name__ == "__main__":
    main()
