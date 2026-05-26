"""Held-out test-slice scorer for artifact-only SAPOG evaluations.

This evaluator reads the same method artifacts as the visible train slice but
scores disjoint folds. It replays submitted action sequences against the
tracked SAPOG data, so self-reported reward curves in artifacts cannot
determine held-out scores. When scoring/scores_train.json exists, it enforces
that the held-out report contains exactly the same method entries as train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from data.sapog_data import component_roots, fold_split, load_forest


SLICE = "test"
SLICE_FOLDS = {1, 3, 4}

METHOD_NAMES = {
    "pege_ddb": "pege_ddb",
    "greedy_classifier": "greedy_classifier",
    "greedy_neighbor": "greedy_neighbor",
    "random": "random",
    "fog": "fully_observable_gittins",
}


def discover_artifacts() -> List[Path]:
    return sorted(Path("checkpoints").glob("**/results.json"))


def expand_artifacts(paths: Iterable[str]) -> List[Path]:
    artifacts: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            artifacts.extend(sorted(path.glob("**/results.json")))
        else:
            artifacts.append(path)
    seen: set[Path] = set()
    unique: List[Path] = []
    for artifact in artifacts:
        key = artifact.resolve() if artifact.exists() else artifact
        if key not in seen:
            seen.add(key)
            unique.append(artifact)
    return unique


def discounted_auc(curve: List[float]) -> float:
    if not curve:
        return 0.0
    arr = np.asarray(curve, dtype=float)
    if len(arr) == 1:
        return float(arr[0])
    area = float(np.sum((arr[:-1] + arr[1:]) * 0.5))
    return area / len(arr)


def replay_row(ds, row: dict, gamma: float, budget_fraction: float) -> dict:
    fold = int(row["fold"])
    _, test_indices = fold_split(ds, fold)
    test_set = set(int(i) for i in test_indices)
    frontier = set(component_roots(ds, test_indices))
    tested: set[int] = set()
    rewards: List[float] = []
    valid_tests = 0
    budget = max(1, int(round(len(test_indices) * budget_fraction)))

    for raw_action in row.get("actions", [])[:budget]:
        try:
            action = int(raw_action)
        except (TypeError, ValueError):
            action = -1

        if action in frontier and action in test_set and action not in tested:
            frontier.remove(action)
            tested.add(action)
            valid_tests += 1
            rewards.append((gamma ** len(rewards)) * int(ds.labels[action]))
            for child in ds.children[action]:
                if child in test_set and child not in tested:
                    frontier.add(int(child))
        else:
            rewards.append(0.0)

    while len(rewards) < budget:
        rewards.append(0.0)

    cumulative = np.cumsum(rewards).astype(float).tolist()
    return {
        "fold": fold,
        "test_nodes": len(test_indices),
        "cumulative_discounted_reward": cumulative,
        "auc": discounted_auc(cumulative),
        "budget_fraction": valid_tests / budget,
    }


def load_required_methods(path: Path) -> Optional[Dict[str, Set[str]]]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    required: Dict[str, Set[str]] = {}
    for experiment, payload in raw.get("experiments", {}).items():
        required[experiment] = set(payload.get("results", {}).keys())
    return required


def weighted_mean(rows: List[dict], key: str) -> float:
    weights = [float(r.get("test_nodes", 1.0)) for r in rows]
    vals = [float(r[key]) for r in rows]
    total = sum(weights) or 1.0
    return sum(w * v for w, v in zip(weights, vals)) / total


def weighted_curve_value(rows: List[dict], fraction: float) -> float:
    weights = [float(r.get("test_nodes", 1.0)) for r in rows]
    vals = []
    for row in rows:
        curve = row.get("cumulative_discounted_reward", [])
        if not curve:
            vals.append(0.0)
            continue
        idx = max(0, min(len(curve) - 1, int(round(len(curve) * fraction)) - 1))
        vals.append(float(curve[idx]))
    total = sum(weights) or 1.0
    return sum(w * v for w, v in zip(weights, vals)) / total


def slice_rows(rows: Iterable[dict]) -> List[dict]:
    return [r for r in rows if int(r.get("fold", -1)) in SLICE_FOLDS]


def summarize_artifact(path: Path) -> Dict[str, Dict[str, float]]:
    raw = json.loads(path.read_text())
    ds = load_forest(raw.get("data", "data/synthetic_hiv_forest.json"))
    gamma = float(raw.get("gamma", 0.99))
    budget_fraction = float(raw.get("budget_fraction", 1.0))
    out: Dict[str, Dict[str, float]] = {}
    for policy, all_rows in raw["policies"].items():
        rows = [replay_row(ds, row, gamma, budget_fraction) for row in slice_rows(all_rows)]
        if not rows:
            continue
        observed_folds = {int(r["fold"]) for r in rows if "fold" in r}
        if observed_folds != SLICE_FOLDS:
            missing = sorted(SLICE_FOLDS - observed_folds)
            raise ValueError(f"{path}: policy {policy} is missing {SLICE} folds {missing}")
        name = METHOD_NAMES.get(policy, policy)
        out[name] = {
            "auc": weighted_mean(rows, "auc"),
            "reward_25pct": weighted_curve_value(rows, 0.25),
            "reward_50pct": weighted_curve_value(rows, 0.50),
            "reward_75pct": weighted_curve_value(rows, 0.75),
            "budget_fraction": weighted_mean(rows, "budget_fraction"),
        }
    return out


def enforce_train_method_set(
    results: Dict[str, Dict[str, Dict[str, float]]],
    required: Optional[Dict[str, Set[str]]],
) -> None:
    if required is None:
        return
    for experiment, methods in required.items():
        if experiment not in results:
            raise ValueError(f"{experiment}: missing from held-out {SLICE} scores")
        observed = set(results[experiment]["results"])
        missing = methods - observed
        if missing:
            raise ValueError(f"{experiment}: held-out {SLICE} scores missing methods {sorted(missing)}")
        extra = observed - methods
        for method in extra:
            del results[experiment]["results"][method]
    for experiment in list(results):
        if experiment not in required:
            del results[experiment]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifact_paths = expand_artifacts(args.artifacts) if args.artifacts else discover_artifacts()
    if not artifact_paths:
        raise FileNotFoundError("no checkpoints/**/results.json artifacts were found")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for artifact in artifact_paths:
        if not artifact.exists():
            continue
        raw = json.loads(artifact.read_text())
        experiment = str(raw.get("experiment", "synthetic_main"))
        results.setdefault(experiment, {"slice": SLICE, "split_folds": sorted(SLICE_FOLDS), "results": {}})
        results[experiment]["results"].update(summarize_artifact(artifact))

    if not results:
        raise FileNotFoundError("no result artifacts were found or readable")
    enforce_train_method_set(results, load_required_methods(Path(args.train_scores)))
    for experiment, payload in results.items():
        if not payload["results"]:
            raise ValueError(f"{experiment}: no methods had rows for {SLICE} folds {sorted(SLICE_FOLDS)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"slice": SLICE, "experiments": results}, indent=2))
    methods = {exp: sorted(payload["results"]) for exp, payload in results.items()}
    print(json.dumps({"wrote": str(output), "slice": SLICE, "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
