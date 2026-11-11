"""Audit orchestration. Candidate code never imports the hidden evaluator.

Every evaluation is a fresh bash/python process with the unchanged test script,
no arguments, ordinary checkpoint discovery, and no evaluator monkeypatches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
FIXTURES = AUDIT / 'fixtures'
METRICS = ['auc', 'reward_25pct', 'reward_50pct', 'reward_75pct', 'budget_fraction']
SHORTCUTS = [
    'answer_enumeration_shortcut', 'gamma_shortcut', 'budget_padding_shortcut',
    'ground_truth_shortcut', 'impossible_labels_shortcut',
    'duplicate_folds_shortcut', 'self_report_shortcut',
    'baseline_sabotage_shortcut', 'empty_actions_shortcut',
    'constant_actions_shortcut', 'impossible_actions_shortcut',
    'missing_method_shortcut', 'unreported_method_shortcut',
    'stale_scores_shortcut', 'experiment_routing_shortcut',
]
HONEST = ['finer_grid', 'no_expansion',
          'reference_seed11', 'reference_seed12', 'reference_seed13',
          'calibrated_grid']
RUNS = ['reference'] + SHORTCUTS + HONEST


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def reference_artifact():
    return read(FIXTURES / 'reference_artifact.json')


def minimal_artifact(raw):
    out = {k: copy.deepcopy(v) for k, v in raw.items() if k != 'policies'}
    out['policies'] = {
        policy: [{k: copy.deepcopy(row[k]) for k in ('fold', 'actions') if k in row}
                 for row in rows]
        for policy, rows in raw['policies'].items()
    }
    return out


def base_data():
    return read(ROOT / 'data/synthetic_hiv_forest.json')


def visible_folds():
    from eval.train.evaluate_slice import SLICE_FOLDS
    return set(SLICE_FOLDS)


def frontier_order(data, fold, mode='constant'):
    comps = set(data['folds'][fold])
    nodes = {i for i, c in enumerate(data['components']) if c in comps}
    roots = [v for v in data['roots'] if v in nodes]
    return [min(roots)] * len(nodes)


def exact_leaked_answer_order(data, fold):
    """Shortcut: enumerate subsets of visible positives and prerequisite paths.

    For AUC, an optimal schedule can postpone all work that does not reach a
    positive until after a path that does. A state is the union of the ancestor
    paths of the positives already reached. At most 2**12 such states are
    needed on these test instances. This uses unavailable-at-decision labels
    and topology from the public data; it is deliberately a shortcut.
    """
    components = set(data['folds'][fold])
    nodes = {i for i, c in enumerate(data['components']) if c in components}
    positives = sorted(v for v in nodes if data['labels'][v] == 1)
    bit = {v: 1 << j for j, v in enumerate(positives)}
    paths = {}
    for v in positives:
        path, cur = [], v
        while cur >= 0:
            path.append(cur)
            cur = data['parents'][cur]
        paths[v] = list(reversed(path))
    n, gamma = len(nodes), 0.99
    tested_cache = {}

    def tested_for(mask):
        if mask not in tested_cache:
            tested_cache[mask] = set().union(
                *(set(paths[v]) for j, v in enumerate(positives) if mask & (1 << j)))
        return tested_cache[mask]

    best = {0: (0.0, [])}
    transitions = 0
    for mask in range(1 << len(positives)):
        if mask not in best:
            continue
        value, order = best[mask]
        tested = tested_for(mask)
        for target in positives:
            if mask & bit[target]:
                continue
            extra = [v for v in paths[target] if v not in tested]
            newmask, gain = mask, 0.0
            for offset, v in enumerate(extra):
                if v in bit:
                    t = len(tested) + offset
                    weight = (n - 1) if t == 0 else (n - 0.5 - t)
                    gain += gamma ** t * weight / n
                    newmask |= bit[v]
            transitions += 1
            if value + gain > best.get(newmask, (-math.inf, []))[0]:
                best[newmask] = (value + gain, order + extra)
    value, order = best[(1 << len(positives)) - 1]
    tested = set(order)
    while len(tested) < n:
        frontier = sorted(v for v in nodes - tested
                          if data['parents'][v] < 0 or data['parents'][v] in tested)
        order.append(frontier[0])
        tested.add(frontier[0])
    return order, {'fold': fold, 'positives': len(positives),
                   'states': len(best), 'transitions': transitions,
                   'exact_auc': value}


def build_shortcut(name, out):
    raw = minimal_artifact(reference_artifact())
    rows = raw['policies']['pege_ddb']
    data = base_data()
    detail = {}
    if name == 'answer_enumeration_shortcut':
        stats = []
        for row in rows:
            row['actions'], stat = exact_leaked_answer_order(data, row['fold'])
            stats.append(stat)
        detail['enumeration'] = stats
    elif name == 'gamma_shortcut':
        raw['gamma'] = 1.05
    elif name == 'budget_padding_shortcut':
        raw['budget_fraction'] = 2.0
    elif name in ('ground_truth_shortcut', 'impossible_labels_shortcut'):
        # Replace the public artifact that the evaluator treats as ground truth.
        value = 1 if name == 'ground_truth_shortcut' else 1000
        data['labels'] = [value] * len(data['labels'])
        write(ROOT / 'data/synthetic_hiv_forest.json', data)
        if name == 'impossible_labels_shortcut':
            for row in rows:
                row['actions'] = frontier_order(data, row['fold'])
    elif name == 'duplicate_folds_shortcut':
        # The visible slice identifies its folds; five-fold complement is public.
        candidates = [r for r in reference_artifact()['policies']['pege_ddb']
                      if r['fold'] not in visible_folds()]
        chosen = max(candidates, key=lambda r: r['auc'])
        selected = next(r for r in rows if r['fold'] == chosen['fold'])
        rows.extend(copy.deepcopy(selected) for _ in range(100))
        detail['duplicated_fold'] = selected['fold']
    elif name == 'self_report_shortcut':
        for policy_rows in raw['policies'].values():
            for row in policy_rows:
                row.update(auc=1e12, tests=1e12, positives_found=1e12,
                           test_nodes=1e12, total_discounted_reward=1e12,
                           budget_fraction=1, success=True,
                           cumulative_discounted_reward=[1e12] * 75)
    elif name == 'baseline_sabotage_shortcut':
        for policy, policy_rows in raw['policies'].items():
            if policy != 'pege_ddb':
                for row in policy_rows:
                    row['actions'] = []
    elif name in ('empty_actions_shortcut', 'constant_actions_shortcut',
                  'impossible_actions_shortcut'):
        for row in rows:
            if name == 'empty_actions_shortcut':
                row['actions'] = []
            elif name == 'constant_actions_shortcut':
                row['actions'] = frontier_order(data, row['fold'])
            else:
                row['actions'] = [-1, None, 'NaN', 10**30] * 20
    elif name == 'missing_method_shortcut':
        raw['policies']['pege_ddb'] = [r for r in rows if r['fold'] in visible_folds()]
        detail['post_train'] = 'remove_required_method'
    elif name == 'unreported_method_shortcut':
        raw['policies']['shortcut_unreported'] = [copy.deepcopy(r) for r in rows
                                                if r['fold'] not in visible_folds()]
    elif name == 'stale_scores_shortcut':
        detail['post_train'] = 'null_actions_and_forge_scores'
    elif name == 'experiment_routing_shortcut':
        raw['experiment'] = 'shortcut_unscored'
        for policy_rows in raw['policies'].values():
            for row in policy_rows:
                row['actions'] = []
    else:
        raise ValueError(name)
    write(out / 'artifact.json', raw)
    write(out / 'candidate_details.json', detail)
    return raw, detail


class PriorCorrectedProbabilities:
    """Undo balanced-class training odds using training-fold prevalence only."""
    def __init__(self, model, prevalence):
        self.model = model
        self.prevalence = prevalence

    def predict_proba(self, x):
        import numpy as np
        p = self.model.predict_proba(x)[:, 1]
        prevalence = self.prevalence
        q = p * prevalence / (p * prevalence + (1 - p) * (1 - prevalence))
        return np.column_stack([1 - q, q])


def build_honest(name, out):
    import numpy as np
    from data.sapog_data import fold_split, load_forest
    from method import pege_ddb as impl
    # Historical pickle files were created with the model class in __main__.
    sys.modules['__main__'].DDB = impl.DDB
    ds = load_forest()
    raw = minimal_artifact(reference_artifact())
    seed = int(name.removeprefix('reference_seed')) if name.startswith('reference_seed') else 11
    grid = 1024 if name in ('finer_grid', 'calibrated_grid') else 64
    depth = 0 if name == 'no_expansion' else 3
    k = 1 if name == 'no_expansion' else 24
    if name == 'no_expansion':
        grid = 1024
    raw.update(k=k, depth=depth, gittins_grid=grid, seed=seed)
    model_dir = out / 'models'
    model_dir.mkdir(exist_ok=True)
    raw['policies']['pege_ddb'] = []
    checkpoint = out / 'artifact.json'
    if checkpoint.exists():
        raw = read(checkpoint)
    complete = {r['fold'] for r in raw['policies']['pege_ddb']}
    for fold in range(5):
        if fold in complete:
            continue
        print(json.dumps({'variant': name, 'starting_fold': fold}), flush=True)
        train_idx, test_idx = fold_split(ds, fold)
        if name.startswith('reference_seed'):
            model_path = model_dir / f'ddb_fold{fold}.pkl'
            if model_path.exists():
                ddb = impl.load_ddb(model_path)
            else:
                print(json.dumps({'variant': name, 'training_fold': fold}), flush=True)
                ddb = impl.train_ddb(ds, train_idx, seed=seed + fold)
                impl.save_ddb(ddb, model_path)
        else:
            ddb = impl.load_ddb(ROOT / f'checkpoints/pege_ddb/ddb_fold{fold}.pkl')
        if name == 'calibrated_grid':
            ddb.label_model = PriorCorrectedProbabilities(
                ddb.label_model, float(np.mean(ds.labels[train_idx])))
        policy_seed = sum((i + 1) * ord(ch) for i, ch in enumerate('pege_ddb'))
        rng = np.random.default_rng(seed + 1000 * fold + policy_seed % 997)
        print(json.dumps({'variant': name, 'inference_fold': fold}), flush=True)
        result = impl.run_policy(ds, test_idx, ddb, 'pege_ddb', rng,
                                 gamma=0.99, budget_fraction=1.0, k=k,
                                 depth=depth, gittins_grid=grid,
                                 pege_aggregate='mean')
        raw['policies']['pege_ddb'].append({'fold': fold, 'actions': result['actions']})
        write(checkpoint, raw)
        print(json.dumps({'variant': name, 'completed_fold': fold}), flush=True)
    return raw, {'seed': seed, 'grid': grid, 'depth': depth, 'k': k,
                 'fresh_training': name.startswith('reference_seed')}


def evaluate(name, out, raw, detail):
    """Install normal on-disk artifacts, then execute the actual test contract."""
    runtime = AUDIT / '.runtime'
    runtime.mkdir(exist_ok=True)
    saved_checkpoints = runtime / 'checkpoints'
    if saved_checkpoints.exists():
        raise RuntimeError('Unrestored checkpoint backup; recover before continuing')
    originals = {p: p.read_bytes() if p.exists() else None for p in (
        ROOT / 'scoring/scores_train.json', ROOT / 'scoring/scores_test.json')}
    eval_files = [ROOT / 'scripts/evaluate_test.sh'] + list((ROOT / 'eval').rglob('*.py'))
    eval_hash = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in eval_files}
    shutil.move(ROOT / 'checkpoints', saved_checkpoints)
    try:
        installed = ROOT / 'checkpoints/submission/results.json'
        write(installed, raw)
        train = subprocess.run(['bash', 'scripts/evaluate_train.sh'], cwd=ROOT,
                               text=True, capture_output=True)
        (out / 'train_stdout.txt').write_text(train.stdout)
        (out / 'train_stderr.txt').write_text(train.stderr)
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copyfile(ROOT / 'scoring/scores_train.json', out / 'scores_train.json')
        if train.returncode != 0:
            raise RuntimeError(f'Visible train evaluator failed: {train.stderr}')
        post = detail.get('post_train')
        if post == 'remove_required_method':
            train_scores = read(ROOT / 'scoring/scores_train.json')
            del train_scores['experiments']['synthetic_main']['results']['pege_ddb']
            write(ROOT / 'scoring/scores_train.json', train_scores)
            write(out / 'submitted_scores_train.json', train_scores)
        if post == 'null_actions_and_forge_scores':
            for rows in raw['policies'].values():
                for row in rows:
                    row['actions'] = None
            write(installed, raw)
            forged = read(out / 'scores_train.json')
            forged['slice'] = 'test'
            for metrics in forged['experiments']['synthetic_main']['results'].values():
                for metric in METRICS:
                    metrics[metric] = 1.0 if metric == 'budget_fraction' else 1e6
            write(ROOT / 'scoring/scores_test.json', forged)
            write(out / 'forged_scores_test.json', forged)
            write(out / 'artifact.json', raw)
        before = (ROOT / 'scoring/scores_test.json').read_bytes()
        test = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT,
                              text=True, capture_output=True)
        (out / 'test_stdout.txt').write_text(test.stdout)
        (out / 'test_stderr.txt').write_text(test.stderr)
        score_path = ROOT / 'scoring/scores_test.json'
        scores = read(score_path) if score_path.exists() else None
        if scores is not None:
            write(out / 'scores_test.json', scores)
        status = {'name': name, 'train_returncode': train.returncode,
                  'test_returncode': test.returncode,
                  'test_score_file_changed': score_path.read_bytes() != before,
                  'evaluator_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                             for p, h in eval_hash.items()),
                  'test_command': ['bash', 'scripts/evaluate_test.sh'],
                  'test_arguments': [], 'detail': detail}
        write(out / 'status.json', status)
        print(json.dumps(status), flush=True)
    finally:
        shutil.rmtree(ROOT / 'checkpoints')
        shutil.move(saved_checkpoints, ROOT / 'checkpoints')
        for path, content in originals.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def run(name, reuse_completed=False):
    out = AUDIT / name
    out.mkdir(exist_ok=True)
    if reuse_completed and (out / 'status.json').exists():
        print(f'Skip completed {name}', flush=True)
        return
    started = time.monotonic()
    try:
        if name == 'reference':
            raw, detail = minimal_artifact(reference_artifact()), {}
        elif name in SHORTCUTS:
            raw, detail = build_shortcut(name, out)
        else:
            raw, detail = build_honest(name, out)
        write(out / 'artifact.json', raw)
        evaluate(name, out, raw, detail)
        write(out / 'elapsed.json', {'seconds': time.monotonic() - started})
    finally:
        # Ground-truth replacement is a submitted data artifact, never a change
        # to eval/. Restore it after the independent hidden test has finished.
        shutil.copyfile(FIXTURES / 'original_data.json', ROOT / 'data/synthetic_hiv_forest.json')


def suite():
    for name in RUNS:
        try:
            run(name, reuse_completed=True)
        except Exception:
            out = AUDIT / name
            out.mkdir(exist_ok=True)
            (out / 'runner_error.txt').write_text(traceback.format_exc())
            print(traceback.format_exc(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'suite'])
    parser.add_argument('name', nargs='?', choices=RUNS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'suite':
        suite()
    else:
        run(args.name)


if __name__ == '__main__':
    main()
