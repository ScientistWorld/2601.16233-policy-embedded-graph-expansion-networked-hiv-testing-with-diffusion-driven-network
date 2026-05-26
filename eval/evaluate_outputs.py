"""Evaluate method artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


METHOD_NAMES = {
    "pege_ddb": "pege_ddb",
    "greedy_classifier": "greedy_classifier",
    "greedy_neighbor": "greedy_neighbor",
    "random": "random",
    "fog": "fully_observable_gittins",
}


def weighted_mean(rows: List[dict], key: str) -> float:
    weights = [float(r.get("test_nodes", 1.0)) for r in rows]
    vals = [float(r[key]) for r in rows]
    total = sum(weights) or 1.0
    return sum(w * v for w, v in zip(weights, vals)) / total


def weighted_curve_value(rows: List[dict], fraction: float) -> float:
    weights = [float(r.get("test_nodes", 1.0)) for r in rows]
    vals = []
    for r in rows:
        curve = r.get("cumulative_discounted_reward", [])
        if not curve:
            vals.append(0.0)
            continue
        idx = max(0, min(len(curve) - 1, int(round(len(curve) * fraction)) - 1))
        vals.append(float(curve[idx]))
    total = sum(weights) or 1.0
    return sum(w * v for w, v in zip(weights, vals)) / total


def summarize_artifact(path: Path) -> Dict[str, Dict[str, float]]:
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    expected_folds = raw.get("folds")
    for policy, rows in raw["policies"].items():
        if not rows:
            continue
        if isinstance(expected_folds, int) and expected_folds > 0:
            observed_folds = {int(r["fold"]) for r in rows if "fold" in r}
            if len(observed_folds) != expected_folds:
                raise ValueError(
                    f"{path}: policy {policy} has {len(observed_folds)} folds, "
                    f"expected {expected_folds}"
                )
        name = METHOD_NAMES.get(policy, policy)
        out[name] = {
            "auc": weighted_mean(rows, "auc"),
            "reward_25pct": weighted_curve_value(rows, 0.25),
            "reward_50pct": weighted_curve_value(rows, 0.50),
            "reward_75pct": weighted_curve_value(rows, 0.75),
            "budget_fraction": float(raw.get("budget_fraction", 1.0)),
        }
    return out


def summarize_relative_k_sweep(raw_artifacts: List[dict]) -> Dict[str, Dict[str, float]]:
    auc_by_variant: Dict[str, float] = {}
    budget_by_variant: Dict[str, float] = {}
    for raw in raw_artifacts:
        expected_folds = raw.get("folds")
        for variant, rows in raw.get("policies", {}).items():
            if not rows:
                continue
            if isinstance(expected_folds, int) and expected_folds > 0:
                observed_folds = {int(r["fold"]) for r in rows if "fold" in r}
                if len(observed_folds) != expected_folds:
                    raise ValueError(
                        f"synthetic_k_sweep variant {variant} has {len(observed_folds)} folds, "
                        f"expected {expected_folds}"
                    )
            auc_by_variant[variant] = weighted_mean(rows, "auc")
            budget_by_variant[variant] = float(raw.get("budget_fraction", 1.0))
    if "pege_k12" not in auc_by_variant:
        raise ValueError("synthetic_k_sweep requires a pege_k12 reference variant")
    base = auc_by_variant["pege_k12"] or 1.0
    return {
        variant: {
            "auc_lift_vs_k12_pct": 100.0 * (auc / base - 1.0),
            "budget_fraction": budget_by_variant.get(variant, 1.0),
        }
        for variant, auc in sorted(auc_by_variant.items())
    }


def summarize_relative_aggregation(raw_artifacts: List[dict]) -> Dict[str, Dict[str, float]]:
    auc_by_variant: Dict[str, float] = {}
    budget_by_variant: Dict[str, float] = {}
    for raw in raw_artifacts:
        expected_folds = raw.get("folds")
        for variant, rows in raw.get("policies", {}).items():
            if not rows:
                continue
            if isinstance(expected_folds, int) and expected_folds > 0:
                observed_folds = {int(r["fold"]) for r in rows if "fold" in r}
                if len(observed_folds) != expected_folds:
                    raise ValueError(
                        f"synthetic_aggregation variant {variant} has {len(observed_folds)} folds, "
                        f"expected {expected_folds}"
                    )
            auc_by_variant[variant] = weighted_mean(rows, "auc")
            budget_by_variant[variant] = float(raw.get("budget_fraction", 1.0))
    if "agg_mean" not in auc_by_variant:
        raise ValueError("synthetic_aggregation requires an agg_mean reference variant")
    base = auc_by_variant["agg_mean"] or 1.0
    return {
        variant: {
            "auc_ratio_to_mean": auc / base,
            "budget_fraction": budget_by_variant.get(variant, 1.0),
        }
        for variant, auc in sorted(auc_by_variant.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=["checkpoints/pege_ddb/results.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    special: Dict[str, List[dict]] = {"synthetic_k_sweep": [], "synthetic_aggregation": []}
    for artifact in args.artifacts:
        p = Path(artifact)
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        experiment = str(raw.get("experiment", "synthetic_main"))
        if experiment in special:
            special[experiment].append(raw)
            continue
        results.setdefault(experiment, {"results": {}})
        results[experiment]["results"].update(summarize_artifact(p))
    if special["synthetic_k_sweep"]:
        results["synthetic_k_sweep"] = {"results": summarize_relative_k_sweep(special["synthetic_k_sweep"])}
    if special["synthetic_aggregation"]:
        results["synthetic_aggregation"] = {"results": summarize_relative_aggregation(special["synthetic_aggregation"])}
    if not results:
        raise FileNotFoundError("no result artifacts were found or readable")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"experiments": results}, indent=2))
    methods = {exp: sorted(payload["results"]) for exp, payload in results.items()}
    print(json.dumps({"wrote": args.output, "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
