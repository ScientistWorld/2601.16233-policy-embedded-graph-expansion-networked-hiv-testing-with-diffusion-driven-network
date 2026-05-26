"""Policy-Embedded Graph Expansion with Dynamics-Driven Branching.

This is a scaled implementation of the paper's computation: DDB learns
node-to-node expansion dynamics with Gaussian Process branch counts and a
conditional denoising diffusion model for child covariates; PEGE samples graph
expansions and embeds them inside a forest oracle at every sequential decision.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.sapog_data import ForestDataset, component_roots, fold_split, load_forest


def discounted_auc(curve: Sequence[float]) -> float:
    if not curve:
        return 0.0
    arr = np.asarray(curve, dtype=float)
    if len(arr) == 1:
        return float(arr[0])
    area = float(np.sum((arr[:-1] + arr[1:]) * 0.5))
    return area / len(arr)


@dataclass
class DDB:
    count_model: object
    diffusion_model: object
    label_model: object
    feature_dim: int
    diffusion_steps: int = 12
    max_children: int = 5

    def _alpha_bar(self, t: np.ndarray) -> np.ndarray:
        beta = np.linspace(0.0008, 0.08, self.diffusion_steps)
        alpha_bar = np.cumprod(1.0 - beta)
        return alpha_bar[t]

    def sample_child_features(self, parent_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=self.feature_dim)
        for t in reversed(range(self.diffusion_steps)):
            t_scaled = np.array([t / max(1, self.diffusion_steps - 1)])
            inp = np.concatenate([parent_x, z, t_scaled])[None, :]
            eps = self.diffusion_model.predict(inp)[0]
            ab = self._alpha_bar(np.array([t]))[0]
            x0 = (z - math.sqrt(max(1e-6, 1.0 - ab)) * eps) / math.sqrt(max(1e-6, ab))
            if t > 0:
                ab_prev = self._alpha_bar(np.array([t - 1]))[0]
                z = math.sqrt(ab_prev) * x0 + math.sqrt(max(1e-6, 1.0 - ab_prev)) * rng.normal(size=self.feature_dim)
            else:
                z = x0
        probs = 1.0 / (1.0 + np.exp(-z))
        return (rng.random(self.feature_dim) < probs).astype(np.float32)

    def sample_count(self, parent_x: np.ndarray, rng: np.random.Generator) -> int:
        mean, std = self.count_model.predict(parent_x[None, :], return_std=True)
        lam = max(0.0, float(rng.normal(float(mean[0]), float(std[0]) + 0.35)))
        return int(np.clip(rng.poisson(lam), 0, self.max_children))

    def label_probability(self, x: np.ndarray) -> float:
        return float(self.label_model.predict_proba(x[None, :])[0, 1])


def train_ddb(ds: ForestDataset, train_indices: Sequence[int], seed: int = 0) -> DDB:
    rng = np.random.default_rng(seed)
    train_set = set(int(i) for i in train_indices)
    parent_rows, counts = [], []
    child_pairs = []
    for i in train_indices:
        child_count = sum(1 for c in ds.children[int(i)] if c in train_set)
        parent_rows.append(ds.features[int(i)])
        counts.append(child_count)
        for child in ds.children[int(i)]:
            if child in train_set:
                child_pairs.append((ds.features[int(i)], ds.features[child]))

    kernel = 1.0 * RBF(length_scale=6.0) + WhiteKernel(noise_level=0.4)
    count_model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True, random_state=seed),
    )
    count_model.fit(np.asarray(parent_rows), np.asarray(counts, dtype=float))

    label_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    label_model.fit(ds.features[list(train_indices)], ds.labels[list(train_indices)])

    # Conditional denoising diffusion training examples: predict injected noise
    # from parent covariates, noised child covariates, and diffusion timestep.
    X_rows, y_rows = [], []
    if not child_pairs:
        child_pairs = [(ds.features[i], ds.features[i]) for i in train_indices]
    for parent_x, child_x in child_pairs:
        child_cont = child_x * 2.0 - 1.0
        for _ in range(5):
            t = int(rng.integers(0, 12))
            eps = rng.normal(size=child_cont.shape)
            beta = np.linspace(0.0008, 0.08, 12)
            ab = np.cumprod(1.0 - beta)[t]
            z = math.sqrt(ab) * child_cont + math.sqrt(1.0 - ab) * eps
            X_rows.append(np.concatenate([parent_x, z, [t / 11.0]]))
            y_rows.append(eps)
    diffusion_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(96, 96), activation="relu", max_iter=220, random_state=seed, early_stopping=True),
    )
    diffusion_model.fit(np.asarray(X_rows), np.asarray(y_rows))
    return DDB(count_model, diffusion_model, label_model, ds.features.shape[1])


def save_ddb(model: DDB, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(model, f)


def load_ddb(path: str | Path) -> DDB:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def oracle_scores(
    frontier: Sequence[int],
    children: Dict[int, List[int]],
    features: Dict[int, np.ndarray],
    known_labels: Dict[int, int],
    ddb: DDB,
    gamma: float,
) -> Dict[int, float]:
    memo: Dict[int, float] = {}

    def subtree_value(v: int) -> float:
        if v in memo:
            return memo[v]
        p = float(known_labels[v]) if v in known_labels else ddb.label_probability(features[v])
        child_vals = sorted((subtree_value(c) for c in children.get(v, [])), reverse=True)
        cont = 0.0
        for rank, val in enumerate(child_vals):
            cont += (gamma ** (rank + 1)) * val
        # Gittins-style priority: immediate expected positive reward plus best
        # discounted continuation available after this referral is tested.
        memo[v] = p + cont
        return memo[v]

    return {int(v): subtree_value(int(v)) for v in frontier}


def expand_from_frontier(
    ds: ForestDataset,
    ddb: DDB,
    discovered: Sequence[int],
    frontier: Sequence[int],
    depth: int,
    rng: np.random.Generator,
) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray]]:
    children = {int(v): [] for v in discovered}
    features = {int(v): ds.features[int(v)] for v in discovered}
    next_id = -1
    queue: List[Tuple[int, int]] = [(int(v), 0) for v in frontier]
    while queue:
        parent, dist = queue.pop(0)
        if dist >= depth:
            continue
        n_child = ddb.sample_count(features[parent], rng)
        for _ in range(n_child):
            child = next_id
            next_id -= 1
            child_x = ddb.sample_child_features(features[parent], rng)
            features[child] = child_x
            children.setdefault(parent, []).append(child)
            children[child] = []
            queue.append((child, dist + 1))
    return children, features


def run_policy(
    ds: ForestDataset,
    test_indices: Sequence[int],
    ddb: Optional[DDB],
    policy: str,
    rng: np.random.Generator,
    gamma: float = 0.99,
    budget_fraction: float = 1.0,
    k: int = 12,
    depth: int = 3,
) -> Dict[str, object]:
    test_set = set(int(i) for i in test_indices)
    discovered = set(component_roots(ds, test_indices))
    frontier = set(discovered)
    tested: set[int] = set()
    known_labels: Dict[int, int] = {}
    parent_tested_positive: Dict[int, bool] = {}
    rewards, actions = [], []
    budget = max(1, int(round(len(test_indices) * budget_fraction)))

    while frontier and len(rewards) < budget:
        fr = sorted(frontier)
        if policy == "random":
            action = int(rng.choice(fr))
        elif policy == "greedy_neighbor":
            candidates = [v for v in fr if parent_tested_positive.get(v, False)]
            action = int(rng.choice(candidates or fr))
        elif policy == "greedy_classifier":
            assert ddb is not None
            probs = [ddb.label_probability(ds.features[v]) for v in fr]
            action = int(fr[int(np.argmax(probs))])
        elif policy == "fog":
            assert ddb is not None
            true_children = {v: [c for c in ds.children[v] if c in test_set and c not in tested] for v in test_set}
            feats = {v: ds.features[v] for v in test_set}
            labels = {v: int(ds.labels[v]) for v in test_set}
            scores = oracle_scores(fr, true_children, feats, labels, ddb, gamma)
            action = max(scores, key=scores.get)
        elif policy == "pege_ddb":
            assert ddb is not None
            aggregate = {v: [] for v in fr}
            for _ in range(k):
                syn_children, syn_features = expand_from_frontier(ds, ddb, sorted(discovered), fr, depth, rng)
                scores = oracle_scores(fr, syn_children, syn_features, known_labels, ddb, gamma)
                for v in fr:
                    aggregate[v].append(scores[v])
            means = {v: float(np.mean(vals)) for v, vals in aggregate.items()}
            action = max(means, key=means.get)
        else:
            raise ValueError(f"unknown policy {policy}")

        tested.add(action)
        frontier.remove(action)
        known_labels[action] = int(ds.labels[action])
        rewards.append((gamma ** (len(rewards))) * int(ds.labels[action]))
        actions.append(action)
        for child in ds.children[action]:
            if child in test_set and child not in tested:
                discovered.add(child)
                frontier.add(child)
                parent_tested_positive[child] = bool(ds.labels[action])

    cumulative = np.cumsum(rewards).astype(float).tolist()
    while len(cumulative) < budget:
        cumulative.append(cumulative[-1] if cumulative else 0.0)
    return {
        "policy": policy,
        "actions": actions,
        "discounted_rewards": rewards,
        "cumulative_discounted_reward": cumulative,
        "total_discounted_reward": float(cumulative[-1] if cumulative else 0.0),
        "auc": discounted_auc(cumulative),
        "positives_found": int(sum(ds.labels[a] for a in actions)),
        "tests": len(actions),
    }


def run_experiment(args: argparse.Namespace) -> None:
    ds = load_forest(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = args.policies.split(",")
    all_results: Dict[str, List[Dict[str, object]]] = {p: [] for p in policies}

    for fold in range(args.folds):
        train_idx, test_idx = fold_split(ds, fold)
        model_path = out_dir / f"ddb_fold{fold}.pkl"
        ddb = train_ddb(ds, train_idx, seed=args.seed + fold)
        save_ddb(ddb, model_path)
        for policy in policies:
            stable_policy_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(policy))
            rng = np.random.default_rng(args.seed + 1000 * fold + stable_policy_seed % 997)
            result = run_policy(ds, test_idx, ddb, policy, rng, args.gamma, args.budget_fraction, args.k, args.depth)
            result["fold"] = fold
            result["test_nodes"] = len(test_idx)
            all_results[policy].append(result)

    artifact = {
        "experiment": args.experiment,
        "data": args.data,
        "gamma": args.gamma,
        "k": args.k,
        "depth": args.depth,
        "budget_fraction": args.budget_fraction,
        "folds": args.folds,
        "policies": all_results,
    }
    (out_dir / "results.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps({"wrote": str(out_dir / "results.json"), "policies": policies}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiv_forest.json")
    parser.add_argument("--output-dir", default="checkpoints/pege_ddb")
    parser.add_argument("--experiment", default="synthetic_main")
    parser.add_argument("--policies", default="pege_ddb,greedy_classifier,greedy_neighbor,random,fog")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
