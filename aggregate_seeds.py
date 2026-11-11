"""Export validated mean +/- sample SD over seeds 0/1/2, without pooling questions as runs."""
import json
import math
from pathlib import Path
import statistics

from evaluate import gold_map, load, retrieval_metrics

ROOT = Path('/data/yaokang/ercrag')
OUT = ROOT/'results_pg_seeds'
SPLITS = ['musique_dev', 'hotpotqa_dev', '2wiki_dev', 'open_musique', 'open_hotpotqa', 'open_2wiki']
LABELS = ['MuSiQue', 'HotpotQA', '2Wiki', 'open MuSiQue', 'open HotpotQA', 'open 2Wiki']
METRICS = ['FSR@2', 'FSR@3', 'FSR@4', 'FSR@5', 'SetEM@|G|']


def root(seed):
    return ROOT/'results_pg' if seed == 0 else OUT/f's{seed}'


def summarize(values):
    assert len(values) == 3 and all(math.isfinite(x) for x in values)
    return {'seeds': [0, 1, 2], 'values': values, 'mean': statistics.mean(values), 'sd': statistics.stdev(values)}


def main():
    selected = json.loads((ROOT/'results_pg/g1_selection.json').read_text())['candidate']
    alt = 'pg_cl' if selected == 'ladder' else 'pg_ll'
    ablations = [('pg', r'\method{}'), (alt, 'Alternative candidate reading'), ('pg_cc', 'Proposal-mass lattice'),
                 ('pg_noinc', 'Without incremental training'), ('pg_noopen', 'Without open-style pools'),
                 ('pg_bin', 'Binary target'), ('pg_mc', 'Multiclass target'),
                 ('pg_max', 'Max-pooling decision'), ('pg_v0', 'Single-set decision')]
    result = {'seeds': [0, 1, 2], 'std_definition': 'sample SD across training seeds (ddof=1)',
              'candidate_rule': selected, 'ret': {}, 'paired_vs_beam': {}, 'validation': {}}
    for split in SPLITS:
        methods = ['pg', 'beam', 'ce']
        if split in ('musique_dev', 'open_musique'):
            methods += [k for k, _ in ablations if k not in methods]
        expected = set(gold_map(split))
        for method in methods:
            by_seed = []
            for seed in (0, 1, 2):
                path = root(seed)/f'{split}.{method}.jsonl'
                rows = load(path)
                assert len(rows) == len(expected) and {r['id'] for r in rows} == expected, (path, 'missing/duplicate IDs')
                if method in ('pg', 'pg_bin', 'pg_mc', 'pg_noopen') and seed:
                    assert all(r['search_config']['candidate'] == selected for r in rows), (path, 'configuration drift')
                by_seed.append(retrieval_metrics(split, path))
            result['ret'][f'{split}|{method}'] = {m: summarize([x[m] for x in by_seed]) for m in METRICS}
            result['validation'][f'{split}|{method}'] = {'n_per_seed': len(expected), 'aligned_ids': True}
        result['paired_vs_beam'][split] = {m: summarize([
            a-b for a,b in zip(result['ret'][f'{split}|pg'][m]['values'], result['ret'][f'{split}|beam'][m]['values'])]) for m in METRICS}
    OUT.mkdir(exist_ok=True)
    (OUT/'multiseed_metrics.json').write_text(json.dumps(result, indent=2))

    def cell(split, method, metric):
        x = result['ret'][f'{split}|{method}'][metric]
        return rf"${x['mean']:.1f}\,\pm\,{x['sd']:.1f}$"

    tex = [r'\subsection{Variation across training seeds}\label{sec:seeds}',
           r'We repeat the supervised retrieval comparisons and MuSiQue ablations with training seeds 0, 1 and 2. '
           r'Training questions, hold-outs, epochs and budgets are fixed. The G1 inference configuration selected on the seed-0 source hold-outs is frozen for both additional seeds; no target labels or seed-specific retuning are used. '
           r'Tables~\ref{tab:seeds-main} and~\ref{tab:seeds-abl} report the arithmetic mean and sample standard deviation across the three independently trained runs. '
           r'The question-level bootstrap intervals in the reference experiment measure a different source of variation. QA, stopping diagnostics and latency retain the reference seed-0 protocol.',
           r'\begin{table*}[t]',
           r'\caption{Supervised retrieval over three training seeds: mean $\pm$ sample SD in percentage points. Every run evaluates the same full question set.}\label{tab:seeds-main}',
           r'\centering\small\begin{tabular}{lcccccc}\toprule',
           r'& \multicolumn{2}{c}{Fine-tuned CE} & \multicolumn{2}{c}{Beam Retrieval} & \multicolumn{2}{c}{\method{}}\\',
           r'Setting & FSR@4 & Set-EM & FSR@4 & Set-EM & FSR@4 & Set-EM\\\midrule']
    for split, label in zip(SPLITS, LABELS):
        tex.append(label + ' & ' + ' & '.join(cell(split, k, m) for k in ('ce', 'beam', 'pg') for m in ('FSR@4', 'SetEM@|G|')) + r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table*}', r'\begin{table}[t]',
            r'\caption{MuSiQue ablations over seeds 0, 1 and 2 (mean $\pm$ sample SD). Decision controls reuse each seed\textquotesingle s lattice.}\label{tab:seeds-abl}',
            r'\centering\scriptsize\resizebox{\columnwidth}{!}{\begin{tabular}{lcccc}\toprule',
            r'& \multicolumn{2}{c}{MuSiQue} & \multicolumn{2}{c}{open MuSiQue}\\',
            r'Variant & FSR@4 & Set-EM & FSR@4 & Set-EM\\\midrule']
    for key, label in ablations:
        tex.append(label + ' & ' + ' & '.join(cell(sp, key, m) for sp in ('musique_dev', 'open_musique') for m in ('FSR@4', 'SetEM@|G|')) + r' \\')
    tex += [r'\bottomrule\end{tabular}}\end{table}',
            r'To pair runs by seed, we also compute the Set-EM difference between \method{} and Beam before averaging. The differences (mean $\pm$ sample SD, percentage points) are:']
    comparisons = []
    for split, label in zip(SPLITS, LABELS):
        x = result['paired_vs_beam'][split]['SetEM@|G|']
        comparisons.append(rf"{label} ${x['mean']:+.1f}\pm{x['sd']:.1f}$")
    tex.append('; '.join(comparisons) + '.')
    (OUT/'multiseed_results.tex').write_text('\n'.join(tex) + '\n')
    (OUT/'validation.json').write_text(json.dumps({'seeds': [0, 1, 2], 'retrieval_groups': len(result['validation']),
        'full_aligned_ids': True, 'finite_metrics': True, 'fixed_candidate_rule': selected}, indent=2))
    print('Validated three-seed retrieval and ablations:', len(result['validation']), 'groups.', flush=True)


if __name__ == '__main__':
    main()
