"""Visible train-slice scorer for artifact-only SAPOG evaluations.

This evaluator intentionally scores method output artifacts rather than running
method code. It uses a fixed pair of evaluation folds from the existing
five-fold protocol so future agents can optimize against this visible slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


SLICE = "train"
SLICE_FOLDS = {0, 2}

METHOD_NAMES = {
    "pege_ddb": "pege_ddb",
    "greedy_classifier": "greedy_classifier",
    "greedy_neighbor": "greedy_neighbor",
    "random": "random",
    "fog": "fully_observable_gittins",
}


def discover_artifacts() -> List[Path]:
    return sorted(Path("checkpoints").glob("**/results.json"))


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
    out: Dict[str, Dict[str, float]] = {}
    for policy, all_rows in raw["policies"].items():
        rows = slice_rows(all_rows)
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
            "budget_fraction": float(raw.get("budget_fraction", 1.0)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifact_paths = [Path(p) for p in args.artifacts] if args.artifacts else discover_artifacts()
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
