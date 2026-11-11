"""Fully Observable Gittins (FOG) diagnostic on the visible validation episodes.

The paper's FOG comparator sees the whole forest topology and every
covariate (never unrevealed labels) and tests the frontier person with the
highest Gittins index computed on their true subtree.  It is an unattainable
reference, so it is not a submitted method: it would need information the
evaluator never reveals.  This script runs it in-process on the visible
validation episodes for context (``--out`` JSON).

The theta mapping is the reference DDB's (trained on the training forest);
the paper additionally lets FOG fit theta on the test graph.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sapog_data import load_validation_episodes  # noqa: E402
from method.pege_ddb import Grid, _children_product  # noqa: E402

GAMMA = 0.99


def fog_indices(forest, ddb, grid: Grid) -> Dict[int, float]:
    n = len(forest)
    X = np.asarray(forest.features, dtype=float)
    order: List[int] = []
    stack = list(forest.roots)
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(forest.children[v])
    deriv: Dict[int, np.ndarray] = {}
    index: Dict[int, float] = {}
    for v in reversed(order):  # children before parents
        kids = forest.children[v]
        phi_c = []
        for lab in (0, 1):
            if kids:
                stacked = np.stack([deriv[c][lab] for c in kids])
                prod = _children_product(stacked, np.array([len(kids)]))
            else:
                prod = np.ones((1, grid.m.size))
            phi_c.append(grid.collection(prod)[0] if kids else grid.m.copy())
        par = forest.parents[v]
        par_x = X[par:par + 1] if par >= 0 else np.zeros((1, X.shape[1]))
        d = np.empty((2, grid.m.size))
        for lab in (0, 1):
            q = float(ddb.theta(X[v:v + 1], par_x, np.array([par < 0]), np.array([lab]))[0])
            cont = (1 - q) * GAMMA * phi_c[0] + q * (1 + GAMMA * phi_c[1])
            d[lab] = grid.derivative(np.maximum(grid.m, cont)[None])[0]
        deriv[v] = d
        state = 0 if par < 0 else forest.labels[par]  # referrer's label is revealed before v is testable
        q = float(ddb.theta(X[v:v + 1], par_x, np.array([par < 0]), np.array([state]))[0])
        cont = (1 - q) * GAMMA * phi_c[0] + q * (1 + GAMMA * phi_c[1])
        index[v] = float(grid.index(cont[None])[0])
    return index


def episode_metrics(order_labels: List[int]) -> Dict[str, float]:
    cum, tot = [], 0.0
    for t, y in enumerate(order_labels):
        tot += GAMMA ** t * y
        cum.append(tot)
    n = len(cum)
    auc = sum((cum[i] + cum[i + 1]) * 0.5 for i in range(n - 1)) / n
    out = {"auc": auc}
    for frac, key in ((0.25, "reward_25pct"), (0.5, "reward_50pct"), (0.75, "reward_75pct")):
        out[key] = cum[max(0, min(n - 1, int(round(n * frac)) - 1))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="checkpoints/pege_ddb")
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with (Path(args.model_dir) / "ddb.pkl").open("rb") as f:
        ddb = pickle.load(f)
    ddb.prepare()
    grid = Grid(GAMMA, args.grid)
    rows = []
    for _, forest in load_validation_episodes():
        idx = fog_indices(forest, ddb, grid)
        frontier = list(forest.roots)
        order = []
        while frontier:
            v = max(frontier, key=lambda u: idx[u])
            frontier.remove(v)
            order.append(forest.labels[v])
            frontier.extend(forest.children[v])
        rows.append(episode_metrics(order))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"fully_observable_gittins_validation": agg, "per_episode": rows}, indent=2) + "\n")
    print(json.dumps(agg))


if __name__ == "__main__":
    main()
