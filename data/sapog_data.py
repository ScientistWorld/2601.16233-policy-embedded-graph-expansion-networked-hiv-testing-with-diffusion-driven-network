"""Data utilities for SAPOG network-testing experiments.

The original ICPSR HIV network requires a data-use agreement, so this module
creates a small, tracked substitute with the same structural ingredients needed
by the paper's protocol: forest components, binary node covariates, binary HIV
status, component-level splits, and incrementally revealed referrals.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class ForestDataset:
    features: np.ndarray
    labels: np.ndarray
    parents: np.ndarray
    components: np.ndarray
    roots: List[int]
    children: Dict[int, List[int]]
    folds: List[List[int]]


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_forest(
    path: str | Path = "data/synthetic_hiv_forest.json",
    seed: int = 7,
    n_components: int = 60,
    feature_dim: int = 72,
    max_nodes: int = 620,
) -> Path:
    """Generate a non-trivial HIV-like referral forest.

    The generator deliberately makes labels and branching depend on inherited
    latent risk encoded in binary covariates. Individual covariates carry useful
    information, but positive status also clusters down referral branches; this
    gives network-aware sequential policies room to beat individual-risk
    baselines without making the task easy.
    """

    rng = np.random.default_rng(seed)
    features: List[List[int]] = []
    labels: List[int] = []
    parents: List[int] = []
    components: List[int] = []
    roots: List[int] = []

    for comp in range(n_components):
        if len(features) >= max_nodes:
            break
        comp_risk = rng.normal(0.0, 0.9)
        target_size = int(np.clip(rng.lognormal(mean=2.0, sigma=0.85), 3, 42))
        root = len(features)
        roots.append(root)
        queue: List[Tuple[int, int, float]] = []

        base = rng.binomial(1, 0.16, size=feature_dim)
        base[:8] = rng.binomial(1, _sigmoid(comp_risk + rng.normal(0, 0.4)), size=8)
        root_risk = comp_risk + 0.9 * base[:8].mean() + rng.normal(0, 0.45)
        root_label = int(rng.random() < _sigmoid(-4.05 + 1.15 * root_risk))
        features.append(base.astype(int).tolist())
        labels.append(root_label)
        parents.append(-1)
        components.append(comp)
        queue.append((root, 0, root_risk))

        while queue and len(features) < max_nodes:
            parent, depth, parent_risk = queue.pop(0)
            comp_count = sum(1 for c in components if c == comp)
            if comp_count >= target_size or depth >= 8:
                continue

            parent_x = np.array(features[parent])
            mean_children = _sigmoid(-1.35 + 1.15 * parent_risk + 0.45 * parent_x[8:16].mean())
            # Count distribution is deliberately over-dispersed, as in noisy referral data.
            n_child = int(min(5, rng.poisson(0.40 + 2.7 * mean_children)))
            n_child = min(n_child, target_size - comp_count, max_nodes - len(features))
            for _ in range(n_child):
                child = len(features)
                keep = rng.random(feature_dim) < 0.82
                noise = rng.binomial(1, 0.14, size=feature_dim)
                child_x = np.where(keep, parent_x, noise)
                inherited = 0.75 * parent_risk + 0.55 * child_x[:8].mean() + rng.normal(0, 0.65)
                child_x[:8] = rng.binomial(1, _sigmoid(inherited), size=8)
                child_x[16:24] = rng.binomial(1, _sigmoid(0.6 * inherited - 0.2), size=8)
                label_p = _sigmoid(-4.35 + 1.20 * inherited + 0.50 * labels[parent])
                child_y = int(rng.random() < label_p)
                features.append(child_x.astype(int).tolist())
                labels.append(child_y)
                parents.append(parent)
                components.append(comp)
                queue.append((child, depth + 1, float(inherited)))

    # Balanced fold assignment by component size.
    comp_ids = sorted(set(components))
    comp_sizes = {c: int(np.sum(np.array(components) == c)) for c in comp_ids}
    fold_bins = [[] for _ in range(5)]
    fold_sizes = [0] * 5
    for comp in sorted(comp_ids, key=lambda c: comp_sizes[c], reverse=True):
        j = int(np.argmin(fold_sizes))
        fold_bins[j].append(int(comp))
        fold_sizes[j] += comp_sizes[comp]

    payload = {
        "metadata": {
            "seed": seed,
            "feature_dim": feature_dim,
            "description": "Synthetic forest substitute for the ICPSR HIV network.",
            "positive_rate": float(np.mean(labels)),
        },
        "features": features,
        "labels": labels,
        "parents": parents,
        "components": components,
        "roots": roots,
        "folds": fold_bins,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_forest(path: str | Path = "data/synthetic_hiv_forest.json") -> ForestDataset:
    p = Path(path)
    if not p.exists():
        generate_synthetic_forest(p)
    raw = json.loads(p.read_text())
    parents = np.array(raw["parents"], dtype=int)
    children: Dict[int, List[int]] = {i: [] for i in range(len(parents))}
    for i, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(i)
    return ForestDataset(
        features=np.array(raw["features"], dtype=np.float32),
        labels=np.array(raw["labels"], dtype=np.int64),
        parents=parents,
        components=np.array(raw["components"], dtype=np.int64),
        roots=[int(r) for r in raw["roots"]],
        children=children,
        folds=[[int(c) for c in fold] for fold in raw["folds"]],
    )


def induced_indices(ds: ForestDataset, component_ids: Iterable[int]) -> List[int]:
    comps = set(int(c) for c in component_ids)
    return [i for i, c in enumerate(ds.components) if int(c) in comps]


def fold_split(ds: ForestDataset, fold: int) -> Tuple[List[int], List[int]]:
    test_components = set(ds.folds[fold])
    train_components = set(int(c) for c in ds.components) - test_components
    return induced_indices(ds, train_components), induced_indices(ds, test_components)


def component_roots(ds: ForestDataset, node_indices: Sequence[int]) -> List[int]:
    node_set = set(int(i) for i in node_indices)
    return [r for r in ds.roots if r in node_set]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_hiv_forest.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--components", type=int, default=60)
    parser.add_argument("--max-nodes", type=int, default=620)
    args = parser.parse_args()
    out = generate_synthetic_forest(args.output, args.seed, args.components, 72, args.max_nodes)
    ds = load_forest(out)
    print(f"wrote {out} with {len(ds.labels)} nodes, positive_rate={ds.labels.mean():.3f}")


if __name__ == "__main__":
    main()
