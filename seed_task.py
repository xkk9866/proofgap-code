"""One fixed-protocol training or inference unit for the additional seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/yaokang/ercrag')


def main(seed, ds, key, stage):
    assert seed in (1, 2)
    result = ROOT/f'results_pg_seeds/s{seed}'
    ckpt = ROOT/f'ckpt_pg_seeds/s{seed}/{ds}_{key}'
    result.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RES=str(result), C=str(ckpt.parent))

    def run(*args):
        subprocess.run([sys.executable, *map(str, args)], env=env, check=True)

    if stage == 'train':
        args = ['train.py', '--data', ds, '--out', str(ckpt), '--epochs', '2' if ds == 'musique' else '1',
                '--no_set', '--gc', '--open_mix', '0' if key == 'pg_noopen' else '0.5',
                '--n_train', '16', '--holdout', '1000', '--seed', str(seed),
                '--enc', '/data/yaokang/models/dv3/deberta-v3-base']
        if ds != 'musique':
            args += ['--limit', '21000']
        if key == 'beam':
            args += ['--head', 'point', '--onpolicy_beam', '2', '--hop_heads']
        elif key == 'ce':
            args += ['--head', 'point', '--no_state']
        else:
            args += ['--head', 'ladder', '--rungs', '5', '--pad', '5']
            if key != 'pg_noinc':
                args += ['--inc']
            if key in ('pg_bin', 'pg_mc'):
                args += ['--set_target', 'binary' if key == 'pg_bin' else 'multiclass']
        run(*args)
        return
    selected = json.loads((result/'g1_selection.json').read_text())['candidate']
    for split in (f'{ds}_dev', f'open_{ds}'):
        args = ['infer.py', '--ckpt', str(ckpt), '--split', split, '--out', str(result/f'{split}.{key}.jsonl')]
        if key == 'beam':
            args += ['--search', '2', '--no_accum', '--bs', '2']
        elif key == 'ce':
            args += ['--kmax', '6']
        else:
            args += ['--pg', '2', '--bs', '2', '--cand', 'chain' if key == 'pg_noinc' else selected]
            if key == 'pg_bin':
                args += ['--how', 'u0']
        run(*args)
        if key == 'pg':
            # CPU-only decision controls share the same lattice, for every dataset.
            import dump_pg
            dump_pg.R = str(result)
            dump_pg.decision_variants(split)
        if key == 'pg' and ds == 'musique':
            alt = 'pg_cl' if selected == 'ladder' else 'pg_ll'
            for tag, cand, front in ((alt, 'chain' if selected == 'ladder' else 'ladder', 'ladder'),
                                     ('pg_cc', 'chain', 'chain')):
                run('infer.py', '--ckpt', ckpt, '--split', split, '--out', result/f'{split}.{tag}.jsonl',
                    '--pg', '2', '--bs', '2', '--cand', cand, '--front', front)


if __name__ == '__main__':
    main(int(sys.argv[1]), *sys.argv[2:])
