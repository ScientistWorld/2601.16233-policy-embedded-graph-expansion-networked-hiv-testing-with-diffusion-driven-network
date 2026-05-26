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
from sklearn.svm import LinearSVC

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
    transition: Tuple[float, float]
    classifier_model: object | None = None
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
        probs = np.clip((z + 1.0) * 0.5, 0.0, 1.0)
        return (rng.random(self.feature_dim) < probs).astype(np.float32)

    def sample_count(self, parent_x: np.ndarray, rng: np.random.Generator) -> int:
        mean, std = self.count_model.predict(parent_x[None, :], return_std=True)
        draw = rng.normal(float(mean[0]), float(std[0]) + 0.35)
        return int(np.clip(round(draw), 0, self.max_children))

    def label_probability(self, x: np.ndarray) -> float:
        return float(self.label_model.predict_proba(x[None, :])[0, 1])

    def classifier_score(self, x: np.ndarray) -> float:
        model = getattr(self, "classifier_model", None)
        if model is None:
            return self.label_probability(x)
        if hasattr(model, "decision_function"):
            return float(model.decision_function(x[None, :])[0])
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(x[None, :])[0, 1])
        return float(model.predict(x[None, :])[0])


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
    classifier_model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed),
    )
    classifier_model.fit(ds.features[list(train_indices)], ds.labels[list(train_indices)])

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
    child_labels_by_parent = {0: [], 1: []}
    for i in train_indices:
        for child in ds.children[int(i)]:
            if child in train_set:
                child_labels_by_parent[int(ds.labels[int(i)])].append(int(ds.labels[child]))
    global_rate = float(np.mean(ds.labels[list(train_indices)]))
    transition = tuple(
        float(np.mean(child_labels_by_parent[b])) if child_labels_by_parent[b] else global_rate
        for b in (0, 1)
    )
    return DDB(count_model, diffusion_model, label_model, ds.features.shape[1], transition, classifier_model=classifier_model)


def save_ddb(model: DDB, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(model, f)


def load_ddb(path: str | Path) -> DDB:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _phi_derivative(phi: np.ndarray, grid: np.ndarray) -> np.ndarray:
    deriv = np.gradient(phi, grid)
    return np.clip(deriv, 0.0, 1.0)


def _phi_collection(children: Sequence[int], label: int, phi_by_state: Dict[Tuple[int, int], np.ndarray], grid: np.ndarray, upper: float) -> np.ndarray:
    if not children:
        return grid.copy()
    prod = np.ones_like(grid)
    for child in children:
        prod *= _phi_derivative(phi_by_state[(int(child), int(label))], grid)
    # Phi(m) = upper - integral_m^upper prod(k) dk, evaluated numerically.
    suffix = np.zeros_like(grid)
    for i in range(len(grid) - 2, -1, -1):
        dx = grid[i + 1] - grid[i]
        suffix[i] = suffix[i + 1] + 0.5 * dx * (prod[i] + prod[i + 1])
    return upper - suffix


def oracle_scores(
    frontier: Sequence[int],
    children: Dict[int, List[int]],
    features: Dict[int, np.ndarray],
    known_labels: Dict[int, int],
    ddb: DDB,
    gamma: float,
    grid_size: int = 96,
    parent_label_by_node: Optional[Dict[int, int]] = None,
) -> Dict[int, float]:
    """Numerically compute binary branching-bandit Gittins indices.

    This implements the recursive retirement-value equations used by the
    Gittins oracle in Choo et al. (2025), discretized on a fixed grid. The
    required label-conditionals are provided by the fitted DDB label dynamics.
    """

    upper = 1.0 / max(1e-6, 1.0 - gamma)
    grid = np.linspace(0.0, upper, int(grid_size))
    parent_label_by_node = parent_label_by_node or {}
    nodes = list(features)
    depth_memo: Dict[int, int] = {}

    def depth(v: int) -> int:
        if v not in depth_memo:
            depth_memo[v] = 0 if not children.get(v) else 1 + max(depth(c) for c in children[v])
        return depth_memo[v]

    phi_by_state: Dict[Tuple[int, int], np.ndarray] = {}
    index_by_state: Dict[Tuple[int, int], float] = {}
    for node in sorted(nodes, key=depth):
        for parent_label in (0, 1):
            if node in known_labels:
                p1 = float(known_labels[node])
            elif node < 0:
                p1 = float(ddb.transition[parent_label])
            else:
                base = ddb.label_probability(features[node])
                p1 = float(np.clip(0.65 * base + 0.35 * ddb.transition[parent_label], 1e-4, 1.0 - 1e-4))
            p = {0: 1.0 - p1, 1: p1}
            cont = np.zeros_like(grid)
            for label in (0, 1):
                phi_children = _phi_collection(children.get(node, []), label, phi_by_state, grid, upper)
                cont += p[label] * (float(label) + gamma * phi_children)
            phi = np.maximum(grid, cont)
            phi_by_state[(node, parent_label)] = phi
            viable = np.flatnonzero(cont >= grid)
            index_by_state[(node, parent_label)] = float(grid[int(viable[-1])] if len(viable) else 0.0)

    scores: Dict[int, float] = {}
    for v in frontier:
        parent_label = parent_label_by_node.get(int(v))
        if parent_label in (0, 1):
            scores[int(v)] = index_by_state[(int(v), int(parent_label))]
        else:
            scores[int(v)] = 0.5 * (index_by_state[(int(v), 0)] + index_by_state[(int(v), 1)])
    return scores


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
    gittins_grid: int = 96,
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
            scores = [ddb.classifier_score(ds.features[v]) for v in fr]
            action = int(fr[int(np.argmax(scores))])
        elif policy == "fog":
            assert ddb is not None
            true_children = {v: [c for c in ds.children[v] if c in test_set and c not in tested] for v in test_set}
            feats = {v: ds.features[v] for v in test_set}
            labels = {v: int(ds.labels[v]) for v in test_set}
            parent_context = {
                int(v): int(ds.labels[int(ds.parents[int(v)])])
                for v in fr
                if int(ds.parents[int(v)]) in test_set
            }
            scores = oracle_scores(fr, true_children, feats, labels, ddb, gamma, gittins_grid, parent_context)
            action = max(scores, key=scores.get)
        elif policy == "pege_ddb":
            assert ddb is not None
            aggregate = {v: [] for v in fr}
            parent_context = {
                int(v): int(known_labels[int(ds.parents[int(v)])])
                for v in fr
                if int(ds.parents[int(v)]) in known_labels
            }
            for _ in range(k):
                syn_children, syn_features = expand_from_frontier(ds, ddb, sorted(discovered), fr, depth, rng)
                scores = oracle_scores(fr, syn_children, syn_features, known_labels, ddb, gamma, gittins_grid, parent_context)
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
    artifact_path = out_dir / "results.json"

    all_results: Dict[str, List[Dict[str, object]]] = {p: [] for p in policies}
    if args.resume and artifact_path.exists():
        raw = json.loads(artifact_path.read_text())
        if (
            raw.get("data") == args.data
            and float(raw.get("gamma", -1)) == args.gamma
            and int(raw.get("k", -1)) == args.k
            and int(raw.get("depth", -1)) == args.depth
            and int(raw.get("gittins_grid", -1)) == args.gittins_grid
            and float(raw.get("budget_fraction", -1)) == args.budget_fraction
        ):
            for policy in policies:
                all_results[policy] = list(raw.get("policies", {}).get(policy, []))

    def write_artifact() -> None:
        artifact = {
            "experiment": args.experiment,
            "data": args.data,
            "gamma": args.gamma,
            "k": args.k,
            "depth": args.depth,
            "gittins_grid": args.gittins_grid,
            "budget_fraction": args.budget_fraction,
            "folds": args.folds,
            "policies": all_results,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2))

    for fold in range(args.folds):
        finished = {int(r["fold"]) for rows in all_results.values() for r in rows if "fold" in r}
        if fold in finished and all(any(int(r.get("fold", -1)) == fold for r in rows) for rows in all_results.values()):
            print(json.dumps({"skip_completed_fold": fold}, indent=2))
            continue
        train_idx, test_idx = fold_split(ds, fold)
        model_path = out_dir / f"ddb_fold{fold}.pkl"
        if args.resume and model_path.exists():
            ddb = load_ddb(model_path)
        else:
            ddb = train_ddb(ds, train_idx, seed=args.seed + fold)
            save_ddb(ddb, model_path)
        for policy in policies:
            if any(int(r.get("fold", -1)) == fold for r in all_results[policy]):
                continue
            stable_policy_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(policy))
            rng = np.random.default_rng(args.seed + 1000 * fold + stable_policy_seed % 997)
            result = run_policy(
                ds, test_idx, ddb, policy, rng, args.gamma, args.budget_fraction, args.k, args.depth, args.gittins_grid
            )
            result["fold"] = fold
            result["test_nodes"] = len(test_idx)
            all_results[policy].append(result)
            write_artifact()
            print(json.dumps({"completed_fold": fold, "policy": policy}, indent=2))

    write_artifact()
    print(json.dumps({"wrote": str(artifact_path), "policies": policies}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiv_forest.json")
    parser.add_argument("--output-dir", default="checkpoints/pege_ddb")
    parser.add_argument("--experiment", default="synthetic_main")
    parser.add_argument("--policies", default="pege_ddb,greedy_classifier,greedy_neighbor,random,fog")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--gittins-grid", type=int, default=96)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
