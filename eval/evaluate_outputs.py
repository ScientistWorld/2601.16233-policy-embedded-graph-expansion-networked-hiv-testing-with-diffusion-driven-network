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
    for policy, rows in raw["policies"].items():
        if not rows:
            continue
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
    parser.add_argument("--artifacts", nargs="+", default=["checkpoints/pege_ddb/results.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    results: Dict[str, Dict[str, Dict[str, float]]] = {"synthetic_main": {"results": {}}}
    for artifact in args.artifacts:
        p = Path(artifact)
        if not p.exists():
            continue
        results["synthetic_main"]["results"].update(summarize_artifact(p))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"experiments": results}, indent=2))
    print(json.dumps({"wrote": args.output, "methods": sorted(results["synthetic_main"]["results"])}, indent=2))


if __name__ == "__main__":
    main()
