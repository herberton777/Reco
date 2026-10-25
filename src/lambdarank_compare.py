"""Run the existing top-model experiment and export a readable comparison.

python -m src.lambdarank_compare --top 4 --output artifacts/rerank/compare_v1
All arguments are forwarded to lambdarank_top_models run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from . import lambdarank_top_models as experiment


def export_comparison(folder):
    folder = Path(folder)
    analysis = json.loads((folder / 'top_models_analysis.json').read_text())
    variants = analysis['variants']
    # Preserve validation ordering; never use test scores to choose the winner.
    metric = next(key for key in variants[0]['validation'] if key.startswith('ndcg@'))
    reference = next(row for row in variants
                     if row['variant'] == analysis['best_frozen_reference_on_validation'])
    rows = []
    for variant in sorted(variants, key=lambda row: -row['validation'][metric]):
        row = {'variant': variant['variant'], 'type': variant['type']}
        for split in ('validation', 'test'):
            row.update({f'{split}_{key}': value for key, value in variant[split].items()})
        row['test_ndcg_delta_vs_reference'] = variant['test'][metric] - reference['test'][metric]
        frozen = next((base for base in variants
                       if base['variant'] == variant['variant'].replace('lambdarank_', 'frozen_', 1)), None)
        row['test_ndcg_delta_vs_own_base'] = (
            variant['test'][metric] - frozen['test'][metric]
            if variant['type'] == 'lambdarank' and frozen else None)
        rows.append(row)
    with (folder / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# MovieLens 32M LambdaRank comparison', '',
             'Ordering and champion selection use validation only. Test metrics rank the supplied candidate pools.', '',
             '| Variant | Validation ' + metric + ' | Test ' + metric + ' | Test delta vs reference |',
             '| --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['validation_' + metric]:.6f} | {row['test_' + metric]:.6f} | {row['test_ndcg_delta_vs_reference']:+.6f} |")
    lines += ['', 'Selected champion: ' + analysis['selected_champion_on_validation'],
              '', 'Frozen reference: ' + reference['variant'], '',
              'Paired query bootstrap for champion minus reference: `' +
              json.dumps(analysis['measured_test_delta_vs_frozen']) + '`', '',
              'CSV includes all reported metrics and each single-model reranker gain over its own frozen model.',
              'Confidence intervals resample queries and are not user-cluster robust.']
    (folder / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return rows


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    experiment.main(['run', *args])
    if '--dry-run' not in args:
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', default='artifacts/rerank/top_models_v1')
        output, _ = parser.parse_known_args(args)
        export_comparison(output.output)
        print('Comparison:', Path(output.output) / 'comparison.md')


if __name__ == '__main__':
    main()
