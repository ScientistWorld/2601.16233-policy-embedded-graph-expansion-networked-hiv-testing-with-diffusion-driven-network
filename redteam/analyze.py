"""Lightweight, standard-library analysis of saved hidden-test reports.

The later ranking cohort is unknown. These explicit cohorts are examples, not
predictions of the future rank. No evaluator code is imported or executed.
"""
from pathlib import Path
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
METRICS = ('auc', 'reward_25pct', 'reward_50pct', 'reward_75pct', 'budget_fraction')


def read(path):
    return json.loads(path.read_text())


def summarize():
    metadata = read(HERE / 'fixtures/original_reference.json')['experiments']
    frozen = read(HERE / 'fixtures/original_scores_test.json')['experiments']
    records = {}
    for status_path in sorted(HERE.glob('*/status.json')):
        folder = status_path.parent
        status = read(status_path)
        report = read(folder / 'scores_test.json')
        records[folder.name] = {
            'status': status,
            'report': report,
            'metrics': report.get('experiments', {}).get('synthetic_main', {}).get('results', {}).get('pege_ddb'),
        }
    successful = [n for n, r in records.items() if r['status']['test_returncode'] == 0]
    honest = ['reference', 'finer_grid', 'calibrated_grid', 'no_expansion',
              'reference_seed11', 'reference_seed12', 'reference_seed13']

    def values_for(name, experiment, metric):
        metrics = records[name]['report'].get('experiments', {}).get(experiment, {}).get('results', {}).get('pege_ddb', {})
        value = metrics.get(metric)
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None

    def rank_cohort(names):
        scales = {}
        for experiment, payload in metadata.items():
            for metric, spec in payload['metrics'].items():
                if not abs(spec['coefficient']):
                    continue
                values = [v for n in names if (v := values_for(n, experiment, metric)) is not None]
                scales[f'{experiment}/{metric}'] = statistics.pstdev(values) if values else 0.0
        ranks = {}
        terms = {}
        for name in names:
            terms[name] = {}
            for experiment, payload in metadata.items():
                for metric, spec in payload['metrics'].items():
                    coefficient = abs(spec['coefficient'])
                    if not coefficient:
                        continue
                    value = values_for(name, experiment, metric)
                    reference = frozen.get(experiment, {}).get('results', {}).get('pege_ddb', {}).get(metric)
                    scale = scales[f'{experiment}/{metric}']
                    if value is None or reference is None or scale == 0:
                        continue
                    direction = 1 if spec['higher_is_better'] else -1
                    terms[name][f'{experiment}/{metric}'] = payload['weight'] * coefficient * direction * (value - reference) / scale
            ranks[name] = sum(terms[name].values())
        return {'members': names, 'population_standard_deviations': scales,
                'rank_improvements': ranks, 'metric_contributions': terms}

    seed_names = ['reference_seed11', 'reference_seed12', 'reference_seed13']
    seed_stats = {}
    for metric in METRICS:
        values = [records[n]['metrics'][metric] for n in seed_names]
        seed_stats[metric] = {'values': values, 'min': min(values), 'max': max(values),
                              'range': max(values) - min(values),
                              'population_sd': statistics.pstdev(values)}
    honest_rank = rank_cohort(honest)
    seed_ranks = [honest_rank['rank_improvements'][n] for n in seed_names]
    output = {
        'reference': frozen['synthetic_main']['results']['pege_ddb'],
        'metrics': list(METRICS),
        'records': records,
        'seed_stats': seed_stats,
        'honest_cohort': honest_rank,
        'honest_cohort_seed_rank_range': max(seed_ranks) - min(seed_ranks),
        'all_successful_cohort': rank_cohort(successful),
        'budget_padding_pair_cohort': rank_cohort(['reference', 'budget_padding_shortcut']),
        'impossible_labels_pair_cohort': rank_cohort(['reference', 'impossible_labels_shortcut']),
        'notes': [
            'Population standard deviations; a sample-SD convention rescales a fixed complete cohort without changing its ordering.',
            'Metrics absent from results are skipped, as specified in the audit task.',
            'An absent method may be rejected by an external eligibility check; that check is not supplied here.',
            'The failed stale-scores run is excluded from every successful-run cohort.',
            'Equal zero-variance metrics contribute zero; all honest variants have budget_fraction=1.',
        ],
    }
    (HERE / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    for name, record in records.items():
        metrics = record['metrics']
        vals = 'MISSING' if metrics is None else ', '.join(f'{metrics[m]:.6f}' for m in METRICS)
        print(f'{name}: [{vals}]; exit={record["status"]["test_returncode"]}')
    for name in ('honest_cohort', 'all_successful_cohort', 'budget_padding_pair_cohort'):
        print(name, json.dumps(output[name]['rank_improvements'], sort_keys=True))
    print('honest_cohort_seed_rank_range', output['honest_cohort_seed_rank_range'])
    print('seed_stats', json.dumps(seed_stats, sort_keys=True))


if __name__ == '__main__':
    summarize()
