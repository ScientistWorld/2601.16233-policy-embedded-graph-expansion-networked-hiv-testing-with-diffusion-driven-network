"""Run paper baseline policies through the same artifact format as PEGE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.sapog_data import fold_split, load_forest
from method.pege_ddb import run_policy, train_ddb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiv_forest.json")
    parser.add_argument("--output-dir", default="checkpoints/baselines")
    parser.add_argument("--policies", default="random,greedy_neighbor,greedy_classifier")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    ds = load_forest(args.data)
    policies = args.policies.split(",")
    out = {p: [] for p in policies}
    for fold in range(args.folds):
        train_idx, test_idx = fold_split(ds, fold)
        ddb = train_ddb(ds, train_idx, seed=args.seed + fold)
        for policy in policies:
            rng = np.random.default_rng(args.seed + 2000 * fold + hash(policy) % 997)
            result = run_policy(ds, test_idx, ddb, policy, rng, args.gamma, args.budget_fraction)
            result["fold"] = fold
            result["test_nodes"] = len(test_idx)
            out[policy].append(result)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "experiment": "synthetic_main",
        "data": args.data,
        "gamma": args.gamma,
        "budget_fraction": args.budget_fraction,
        "policies": out,
    }
    (out_dir / "results.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps({"wrote": str(out_dir / "results.json"), "policies": policies}, indent=2))


if __name__ == "__main__":
    main()
