"""If G1 selected the fallback, align any early ablations before exporting tables."""
import json
import os
from pathlib import Path
import subprocess
import sys


def align():
    root = Path(os.environ['RES'])
    selection = root / 'g1_selection.json'
    if not selection.exists() or json.loads(selection.read_text())['candidate'] != 'chain':
        return
    changed = False
    for key in ('pg_noopen', 'pg_mc', 'pg_bin'):
        for split in ('musique_dev', 'open_musique'):
            path = root / f'{split}.{key}.jsonl'
            with path.open() as f:
                first = json.loads(next(f))
            if first.get('search_config', {}).get('candidate') == 'chain':
                continue
            print('Aligning ablation with source-selected configuration:', split, key, flush=True)
            command = [sys.executable, 'infer.py', '--ckpt', f"{os.environ['C']}/musique_{key}",
                       '--split', split, '--out', str(path), '--pg', '2', '--bs', '2', '--cand', 'chain']
            if key == 'pg_bin':
                command += ['--how', 'u0']
            subprocess.run(command, check=True)
            changed = True
    if changed:
        subprocess.run([sys.executable, 'dump_pg.py'], check=True)


if __name__ == '__main__':
    align()
