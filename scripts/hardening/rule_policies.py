"""Hand-written one-step-lookahead rules (hardening probes, never shipped as methods).

A short closed-form policy a future agent could copy: test the frontier person
with the highest

    score(v) = p(v) + w * gamma * c(v) * (p(v) * q1 + (1 - p(v)) * q0)

where p(v) is a logistic P(HIV+ | covariates, referrer's label [, referrer's
covariates]), c(v) a ridge regression of the number of referrals on the
covariates, and q1 / q0 the training-forest positivity of people referred by a
positive / negative referrer.  ``w`` is either fixed or tuned on the visible
validation episodes.  If such a rule outranks the reference method, the test
does not reward the paper's contribution.

    python scripts/hardening/rule_policies.py build --out DIR --kind parent_label --w 1.0
    python scripts/hardening/rule_policies.py build --out DIR --kind parent_features --tune
    python scripts/hardening/rule_policies.py serve --model-dir DIR
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sapog_data import load_train_forest, load_validation_episodes  # noqa: E402
from data.sapog_protocol import EpisodeState, serve  # noqa: E402

GAMMA = 0.99
W_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def feats(kind, X, PX, is_root, ppos):
    cols = [X, is_root[:, None].astype(float), ppos[:, None].astype(float)]
    if kind == "parent_features":
        cols.insert(1, PX)
    return np.hstack(cols)


def forest_inputs(forest):
    X = np.asarray(forest.features, dtype=float)
    y = np.asarray(forest.labels)
    par = np.asarray(forest.parents)
    is_root = par < 0
    PX = np.where(is_root[:, None], 0.0, X[np.maximum(par, 0)])
    ppos = np.where(is_root, 0, y[np.maximum(par, 0)])
    return X, y, par, is_root, PX, ppos


def fit(kind):
    from sklearn.linear_model import LogisticRegression, Ridge

    forest = load_train_forest()
    X, y, par, is_root, PX, ppos = forest_inputs(forest)
    theta = LogisticRegression(C=1.0, max_iter=5000).fit(feats(kind, X, PX, is_root, ppos), y)
    kids = np.bincount(par[par >= 0], minlength=len(y)).astype(float)
    ridge = Ridge(alpha=5.0).fit(X, kids)
    child = ~is_root
    q1 = float(y[child & (ppos == 1)].mean())
    q0 = float(y[child & (ppos == 0)].mean())
    return {"kind": kind, "theta": theta, "ridge": ridge, "q1": q1, "q0": q0}


def scores(model, w, X, PX, is_root, ppos):
    p = model["theta"].predict_proba(feats(model["kind"], X, PX, is_root, ppos))[:, 1]
    c = np.maximum(0.0, model["ridge"].predict(X))
    return p + w * GAMMA * c * (p * model["q1"] + (1 - p) * model["q0"])


def simulate(forest, s):
    frontier = list(forest.roots)
    cum, tot, t = [], 0.0, 0
    while frontier:
        v = max(frontier, key=lambda u: (s[u], -u))
        frontier.remove(v)
        tot += GAMMA ** t * forest.labels[v]
        cum.append(tot)
        frontier.extend(forest.children[v])
        t += 1
    n = len(cum)
    return sum((cum[i] + cum[i + 1]) * 0.5 for i in range(n - 1)) / n


def tune(model):
    eps = [f for _, f in load_validation_episodes()]
    inputs = [forest_inputs(f) for f in eps]
    best = None
    table = {}
    for w in W_GRID:
        aucs = []
        for f, (X, y, par, is_root, PX, ppos) in zip(eps, inputs):
            aucs.append((len(y), simulate(f, scores(model, w, X, PX, is_root, ppos))))
        val = sum(n * a for n, a in aucs) / sum(n for n, _ in aucs)
        table[w] = val
        if best is None or val > table[best]:
            best = w
    return best, table


def cmd_build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = fit(args.kind)
    w, table = (tune(model) if args.tune else (args.w, {}))
    with (out / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    (out / "config.json").write_text(json.dumps({"kind": args.kind, "w": w, "validation_auc_by_w": table,
                                                  "q1": model["q1"], "q0": model["q0"]}, indent=2) + "\n")
    cmd = ["python", "scripts/hardening/rule_policies.py", "serve", "--model-dir", str(out)]
    (out / "policy.json").write_text(json.dumps({"command": cmd}) + "\n")
    print(json.dumps({"rule": str(out), "kind": args.kind, "w": w, "validation_auc_by_w": table}))


def cmd_serve(args):
    d = Path(args.model_dir)
    cfg = json.loads((d / "config.json").read_text())
    with (d / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    w = float(cfg["w"])
    cache = {}

    def choose(state: EpisodeState) -> int:
        new = [v for v in state.frontier if v not in cache]
        if new:
            D = len(state.features[new[0]])
            X = np.asarray([state.features[v] for v in new], dtype=float)
            PX = np.zeros((len(new), D))
            is_root = np.zeros(len(new), dtype=bool)
            ppos = np.zeros(len(new), dtype=int)
            for i, v in enumerate(new):
                par = state.parent.get(v)
                if par is None:
                    is_root[i] = True
                else:
                    PX[i] = state.features[par]
                    ppos[i] = int(state.labels.get(par, 0))
            for v, s in zip(new, scores(model, w, X, PX, is_root, ppos)):
                cache[v] = float(s)
        return max(state.frontier, key=lambda v: cache[v])

    def on_reset(state: EpisodeState) -> None:
        cache.clear()

    serve(choose, on_reset)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--kind", choices=["parent_label", "parent_features"], default="parent_label")
    b.add_argument("--w", type=float, default=1.0)
    b.add_argument("--tune", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    {"build": cmd_build, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()
