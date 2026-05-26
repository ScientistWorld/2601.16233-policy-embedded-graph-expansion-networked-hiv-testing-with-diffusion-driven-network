"""Run PEGE+DDB ablations matching the paper's Figure 6 settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from data.sapog_data import fold_split, load_forest
from method.pege_ddb import load_ddb, run_policy, save_ddb, train_ddb


K_SWEEP = [
    {"name": "pege_k12", "k": 12, "aggregate": "mean"},
    {"name": "pege_k24", "k": 24, "aggregate": "mean"},
    {"name": "pege_k26", "k": 26, "aggregate": "mean"},
]

AGGREGATION = [
    {"name": "agg_mean", "k": 24, "aggregate": "mean"},
    {"name": "agg_mode", "k": 24, "aggregate": "mode"},
    {"name": "agg_mean_plus_var", "k": 24, "aggregate": "mean_plus_var"},
    {"name": "agg_mean_minus_var", "k": 24, "aggregate": "mean_minus_var"},
]


def stable_seed(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 997


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiv_forest.json")
    parser.add_argument("--output-dir", default="checkpoints/pege_ablation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--gittins-grid", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ds = load_forest(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "synthetic_k_sweep": {
            "path": out_dir / "k_sweep.json",
            "variants": K_SWEEP,
            "policies": {cfg["name"]: [] for cfg in K_SWEEP},
        },
        "synthetic_aggregation": {
            "path": out_dir / "aggregation.json",
            "variants": AGGREGATION,
            "policies": {cfg["name"]: [] for cfg in AGGREGATION},
        },
    }

    for experiment, payload in artifacts.items():
        path = payload["path"]
        if args.resume and path.exists():
            raw = json.loads(path.read_text())
            if (
                raw.get("data") == args.data
                and int(raw.get("folds", -1)) == args.folds
                and int(raw.get("depth", -1)) == args.depth
                and int(raw.get("gittins_grid", -1)) == args.gittins_grid
                and float(raw.get("gamma", -1)) == args.gamma
                and float(raw.get("budget_fraction", -1)) == args.budget_fraction
            ):
                payload["policies"] = {
                    cfg["name"]: list(raw.get("policies", {}).get(cfg["name"], []))
                    for cfg in payload["variants"]
                }

    def write_artifact(experiment: str) -> None:
        payload = artifacts[experiment]
        artifact = {
            "experiment": experiment,
            "data": args.data,
            "gamma": args.gamma,
            "depth": args.depth,
            "gittins_grid": args.gittins_grid,
            "budget_fraction": args.budget_fraction,
            "folds": args.folds,
            "variants": payload["variants"],
            "policies": payload["policies"],
        }
        payload["path"].write_text(json.dumps(artifact, indent=2))

    for fold in range(args.folds):
        train_idx, test_idx = fold_split(ds, fold)
        model_path = out_dir / f"ddb_fold{fold}.pkl"
        if args.resume and model_path.exists():
            ddb = load_ddb(model_path)
        else:
            ddb = train_ddb(ds, train_idx, seed=args.seed + fold)
            save_ddb(ddb, model_path)

        for experiment, payload in artifacts.items():
            for cfg in payload["variants"]:
                rows: List[dict] = payload["policies"][cfg["name"]]
                if any(int(r.get("fold", -1)) == fold for r in rows):
                    continue
                rng = np.random.default_rng(args.seed + 3000 * fold + stable_seed(cfg["name"]))
                result = run_policy(
                    ds,
                    test_idx,
                    ddb,
                    "pege_ddb",
                    rng,
                    args.gamma,
                    args.budget_fraction,
                    int(cfg["k"]),
                    args.depth,
                    args.gittins_grid,
                    str(cfg["aggregate"]),
                )
                result["fold"] = fold
                result["test_nodes"] = len(test_idx)
                result["variant"] = cfg["name"]
                result["k"] = int(cfg["k"])
                result["pege_aggregate"] = str(cfg["aggregate"])
                rows.append(result)
                write_artifact(experiment)
                print(json.dumps({"completed_fold": fold, "experiment": experiment, "variant": cfg["name"]}, indent=2))

    for experiment in artifacts:
        write_artifact(experiment)
        print(json.dumps({"wrote": str(artifacts[experiment]["path"])}, indent=2))


if __name__ == "__main__":
    main()
