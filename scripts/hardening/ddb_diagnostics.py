"""DDB diagnostics for HARDENING.md: are generated children valid, informative people?

Compares the previous reference DDB (old 12-step schedule whose last
alpha_bar is ~0.61, re-noising sampler, independent Bernoulli bits; GPR on
#children) with the fixed one (cosine schedule, ancestral sampling, one answer
per categorical question; GPR on |N(v)|), on the visible validation episodes:

* fraction of generated children that are valid one-hot people, mean #ones
* mean absolute gap between generated and real child bit marginals
* theta's mean P(+) for generated vs real children of positive / negative referrers
* correlation of the GPR's predicted count with the true count
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sapog_data import load_validation_episodes  # noqa: E402
import method.pege_ddb as M  # noqa: E402

CAT = json.loads((ROOT / "data" / "train_forest.json").read_text())["categorical_sizes"]


def old_sample(ddb, parent_X, rng):
    steps = 12
    betas = np.linspace(0.0008, 0.08, steps)
    ab_all = np.cumprod(1.0 - betas)
    R, D = parent_X.shape
    z = rng.standard_normal((R, D))
    for t in reversed(range(steps)):
        eps = ddb._mlp(np.hstack([parent_X, z, np.full((R, 1), t / (steps - 1))]))
        ab = ab_all[t]
        x0 = (z - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab)
        z = math.sqrt(ab_all[t - 1]) * x0 + math.sqrt(1.0 - ab_all[t - 1]) * rng.standard_normal((R, D)) if t > 0 else x0
    return (rng.random((R, D)) < np.clip((z + 1.0) * 0.5, 0.0, 1.0)).astype(float)


def valid_onehot(X):
    ok = np.ones(len(X), dtype=bool)
    s = 0
    for size in CAT:
        ok &= X[:, s:s + size].sum(axis=1) == 1
        s += size
    return ok


def theta_p(ddb, kind, X, PX, lab):
    n = len(X)
    feats = M.theta_features(kind, X, PX, np.zeros(n, dtype=bool), np.full(n, lab))
    w = ddb.theta_model.coef_.ravel()
    return 1.0 / (1.0 + np.exp(-(feats @ w + ddb.theta_model.intercept_[0])))


def diagnose(ddb, sampler, kind, count_target):
    rng = np.random.default_rng(0)
    PXs, CXs, plab, deg_true, X_all = [], [], [], [], []
    for _, f in load_validation_episodes():
        X = np.asarray(f.features, dtype=float)
        for v, p in enumerate(f.parents):
            n_kids = len(f.children[v])
            deg_true.append(n_kids + (1 if p >= 0 else 0) if count_target == "degree" else n_kids)
            X_all.append(X[v])
            if p >= 0:
                PXs.append(X[p]); CXs.append(X[v]); plab.append(f.labels[p])
    PX, CX, plab = np.asarray(PXs), np.asarray(CXs), np.asarray(plab)
    gen = sampler(ddb, PX, rng)
    out = {
        "generated_valid_onehot_frac": float(valid_onehot(gen).mean()),
        "generated_mean_ones": float(gen.sum(axis=1).mean()),
        "real_mean_ones": float(CX.sum(axis=1).mean()),
        "bit_marginal_mean_abs_gap": float(np.abs(gen.mean(axis=0) - CX.mean(axis=0)).mean()),
    }
    for lab in (0, 1):
        m = plab == lab
        out[f"theta_mean_generated_parent{lab}"] = float(theta_p(ddb, kind, gen[m], PX[m], lab).mean())
        out[f"theta_mean_real_parent{lab}"] = float(theta_p(ddb, kind, CX[m], PX[m], lab).mean())
    mean, _ = ddb.count_mean_std(np.asarray(X_all))
    out["count_target"] = count_target
    out["count_corr_validation"] = float(np.corrcoef(mean, np.asarray(deg_true, dtype=float))[0, 1])
    out["count_mean_pred"] = float(mean.mean())
    out["count_mean_true"] = float(np.mean(deg_true))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None, help="pickled previous reference DDB")
    ap.add_argument("--new", default="checkpoints/pege_ddb/ddb.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    if args.old and Path(args.old).exists():
        with open(args.old, "rb") as f:
            old = pickle.load(f)
        old.cat_sizes = CAT
        old._prepared = False
        old.prepare()
        res["previous_ddb"] = diagnose(old, old_sample, old.theta_kind, "children")
    with open(args.new, "rb") as f:
        new = pickle.load(f)
    new.prepare()
    res["fixed_ddb"] = diagnose(new, lambda d, px, rng: d.sample_child_features(px, rng), new.theta_kind, "degree")
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
