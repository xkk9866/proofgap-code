"""Execute the preregistered G1 rule using only source training hold-outs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate import gold_map, load, retrieval_metrics


def main():
    root = Path(os.environ['RES'])
    metrics = {}
    saved = root / 'g1_selection.json'
    for split in ('musique_hold', 'musique_hold_open'):
        metrics[split] = {}
        for key in ('pg', 'pg_cl', 'pg_cc'):
            path = root / f'{split}.{key}.jsonl'
            if key == 'pg' and (root / f'{split}.pg_ll.jsonl').exists():
                path = root / f'{split}.pg_ll.jsonl'
            rows = load(path)
            assert len(rows) == len(gold_map(split)) == len({r['id'] for r in rows})
            metrics[split][key] = retrieval_metrics(split, path)
    selected = 'chain' if all(
        m['pg_cl']['SetEM@|G|'] > m['pg']['SetEM@|G|'] + 1.0
        for m in metrics.values()) else 'ladder'
    decision = {'candidate': selected, 'frontier': 'ladder', 'seed': 0,
                'rule': 'chain only if >1 Set-EM point better on BOTH source hold-outs',
                'holdout_metrics': metrics}
    if saved.exists():
        decision = json.loads(saved.read_text())
        selected = decision['candidate']
    # Persist the decision before accessing any evaluation output.
    temp = root / 'g1_selection.json.tmp'
    temp.write_text(json.dumps(decision, indent=2))
    temp.replace(root / 'g1_selection.json')
    print(json.dumps(decision, indent=2), flush=True)
    if selected == 'chain':
        for ds in ('musique', 'hotpotqa', '2wiki'):
            for split in (f'{ds}_dev', f'open_{ds}', f'{ds}_hold', f'{ds}_hold_open'):
                dst = root / f'{split}.pg.jsonl'
                backup = root / f'{split}.pg_ll.jsonl'
                if not backup.exists():
                    shutil.copy2(dst, backup)
                src = root / f'{split}.pg_cl.jsonl'
                if not src.exists():
                    subprocess.run([sys.executable, 'infer.py', '--ckpt', f"{os.environ['C']}/{ds}_pg",
                                    '--split', split, '--out', str(src), '--pg', '2', '--bs', '2',
                                    '--cand', 'chain', '--front', 'ladder'], check=True)
                shutil.copy2(src, dst)


if __name__ == '__main__':
    main()
