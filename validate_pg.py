"""Check required paper artifacts, sample alignment and finite metrics (CPU only)."""
import json
import math
import os
from pathlib import Path
import random

from evaluate import gold_map, load
from gen_latex4 import BASE, DIST, OPEN


def main():
    root = Path(os.environ['RES'])
    counts = {}
    for split in DIST + OPEN:
        gold = gold_map(split)
        ids = set(gold)
        for method in BASE + ['pg']:
            rows = load(root / f'{split}.{method}.jsonl')
            assert len(rows) == len(ids) and {r['id'] for r in rows} == ids, (split, method, 'incomplete retrieval output')
            counts[f'{split}|{method}'] = len(rows)
        expected = set(random.Random(0).sample(list(gold), min(1000, len(gold))))
        for method in ('closed', 'rerank', 'llmset', 'ircot', 'beam', 'pg', 'pg_adapt', 'gold'):
            rows = load(root / f'qa_{split}.{method}_r.jsonl')
            assert len(rows) == len(expected) and {r['id'] for r in rows} == expected, (split, method, 'QA sample mismatch')
    metrics = json.loads((root / 'all_metrics.json').read_text())

    def check(value, path='root'):
        if isinstance(value, dict):
            assert 'error' not in value, (path, value)
            for k, v in value.items():
                check(v, path + '/' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                check(v, path + '/' + str(i))
        elif isinstance(value, float):
            assert math.isfinite(value), (path, value)
    check(metrics)
    report = {'retrieval_counts': counts, 'qa_random_seed': 0, 'qa_sample_alignment': True,
              'finite_metrics': True, 'g1_selection': metrics['g1_selection']}
    (root / 'validation.json').write_text(json.dumps(report, indent=2))
    print('Validated all 54 main-table retrieval files and 48 aligned QA files.', flush=True)


if __name__ == '__main__':
    main()
