"""Policy-Embedded Graph Expansion (PEGE) with Dynamics-Driven Branching (DDB).

Paper: "Policy-Embedded Graph Expansion: Networked HIV Testing with
Diffusion-Driven Network Samples" (Algorithm 1 and Section 5).

* DDB is trained only on parent->child dynamics of the training forest:
  a Gaussian Process regressor gives a Gaussian over a person's number of
  neighbours |N(v)| given their covariates (Section 5; a non-root's successors
  are |N(v)| - 1 because one neighbour is its referrer), and a conditional
  denoising diffusion model (epsilon-prediction MLP, cosine noise schedule,
  ancestral sampling) samples a child's covariates given the parent's
  covariates; each sample is decoded to one valid answer per categorical
  question.  Recursive sampling from each frontier person up to depth ``d``
  produces forest-preserving graph expansions.
* The decision oracle is the Gittins index policy for forests (Choo et al.
  2025): retirement-value recursion with Whittle's product formula for the
  children of a node, conditioning each child on its parent's label.  Its
  feature mapping theta gives P(HIV+ | own covariates, referrer's covariates,
  referrer's label) -- Choo et al.'s pairwise potentials use the covariates of
  both endpoints of an edge.
* PEGE samples ``k`` expansions at every step, evaluates every frontier
  person with the oracle on each sample, averages (mean aggregation) and
  tests the argmax.

Subcommands::

    python method/pege_ddb.py train --out checkpoints/pege_ddb [--k 24 --depth 3 ...]
    python method/pege_ddb.py variant --base checkpoints/pege_ddb --out DIR [--k ...]
    python method/pege_ddb.py serve --model-dir checkpoints/pege_ddb   # used by policy.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sapog_data import Forest, load_train_forest  # noqa: E402
from data.sapog_protocol import EpisodeState, serve  # noqa: E402

DIFF_STEPS = 40
DIFF_REPS = 20  # noisy training examples per parent->child edge


def _cosine_schedule(steps: int, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal cosine schedule; alpha_bar of the last step is ~0, so
    sampling can start from N(0, I) exactly as the forward process ends."""
    f = lambda t: math.cos((t / steps + s) / (1.0 + s) * math.pi / 2.0) ** 2  # noqa: E731
    ab = np.array([f(t + 1) / f(0) for t in range(steps)])
    prev = np.concatenate([[1.0], ab[:-1]])
    return np.clip(1.0 - ab / prev, 1e-5, 0.999)


BETAS = _cosine_schedule(DIFF_STEPS)
ALPHA_BAR = np.cumprod(1.0 - BETAS)

DEFAULT_CONFIG = {
    "k": 24,
    "depth": 3,
    "grid": 256,
    "aggregate": "mean",
    "seed": 11,
    "theta": "parent_features",
    "theta_generated": False,  # variant: separate theta fitted on DDB-generated children
    "oracle_gamma": None,  # variant: planning discount of the oracle (None = the episode's gamma, 0.99)
}


# --------------------------------------------------------------------------- DDB


class DDB:
    """Hybrid GP-count / diffusion-covariate expansion model plus theta."""

    def __init__(self, count_model, diffusion_model, theta_model, feature_dim: int, max_children: int, theta_kind: str,
                 cat_sizes: Sequence[int]):
        self.count_model = count_model  # GPR over |N(v)| (number of neighbours)
        self.diffusion_model = diffusion_model
        self.theta_model = theta_model
        self.feature_dim = feature_dim
        self.max_children = max_children
        self.theta_kind = theta_kind
        self.cat_sizes = [int(c) for c in cat_sizes]
        assert sum(self.cat_sizes) == feature_dim
        self._prepared = False

    # -- fast inference paths (numerically identical to sklearn's predict) --
    def prepare(self) -> None:
        if self._prepared:
            return
        scaler, mlp = self.diffusion_model.steps[0][1], self.diffusion_model.steps[-1][1]
        self._d_mean = scaler.mean_
        self._d_scale = scaler.scale_
        self._d_W = [w.astype(np.float64) for w in mlp.coefs_]
        self._d_b = [b.astype(np.float64) for b in mlp.intercepts_]
        self._t_w = self.theta_model.coef_.ravel().astype(np.float64)
        self._t_b = float(self.theta_model.intercept_[0])
        gen = getattr(self, "theta_gen_model", None)
        self._g_w = gen.coef_.ravel().astype(np.float64) if gen is not None else self._t_w
        self._g_b = float(gen.intercept_[0]) if gen is not None else self._t_b
        rng = np.random.default_rng(0)
        probe = np.hstack([rng.integers(0, 2, (4, 2 * self.feature_dim)).astype(float), rng.random((4, 1))])
        probe[:, self.feature_dim : 2 * self.feature_dim] = rng.normal(size=(4, self.feature_dim))
        assert np.allclose(self._mlp(probe), self.diffusion_model.predict(probe), atol=1e-6)
        self._prepared = True

    def _mlp(self, inp: np.ndarray) -> np.ndarray:
        h = (inp - self._d_mean) / self._d_scale
        n_layers = len(self._d_W)
        for i, (W, b) in enumerate(zip(self._d_W, self._d_b)):
            h = h @ W + b
            if i < n_layers - 1:
                np.maximum(h, 0.0, out=h)
        return h

    def count_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            return np.zeros(0), np.zeros(0)
        mean, std = self.count_model.predict(X, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    def sample_successors(self, mean: np.ndarray, std: np.ndarray, has_referrer: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
        """Sample |N(v)| from the GPR Gaussian; successors exclude the referrer."""
        draw = np.rint(rng.normal(mean, std)) - has_referrer.astype(float)
        return np.clip(draw, 0, self.max_children).astype(np.int64)

    def sample_child_features(self, parent_X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Ancestral DDPM sampling of child covariates given parent covariates."""
        R, D = parent_X.shape[0], self.feature_dim
        if R == 0:
            return np.zeros((0, D))
        z = rng.standard_normal((R, D))
        for t in reversed(range(DIFF_STEPS)):
            inp = np.hstack([parent_X, z, np.full((R, 1), t / (DIFF_STEPS - 1))])
            eps = self._mlp(inp)
            ab = ALPHA_BAR[t]
            x0 = np.clip((z - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab), -1.0, 1.0)
            if t > 0:
                abp, beta = ALPHA_BAR[t - 1], BETAS[t]
                mean = (math.sqrt(abp) * beta / (1.0 - ab)) * x0 + (math.sqrt(1.0 - beta) * (1.0 - abp) / (1.0 - ab)) * z
                var = beta * (1.0 - abp) / (1.0 - ab)
                z = mean + math.sqrt(var) * rng.standard_normal((R, D))
            else:
                z = x0
        # Decode to a valid person: one answer per categorical question.
        weight = np.clip((z + 1.0) * 0.5, 1e-4, None)
        out = np.zeros((R, D))
        u = rng.random((R, len(self.cat_sizes)))
        start = 0
        for j, size in enumerate(self.cat_sizes):
            block = weight[:, start : start + size]
            cdf = np.cumsum(block, axis=1)
            pick = (cdf < (u[:, j] * cdf[:, -1])[:, None]).sum(axis=1)
            out[np.arange(R), start + np.minimum(pick, size - 1)] = 1.0
            start += size
        return out

    def theta(self, X: np.ndarray, parent_X: np.ndarray, is_root: np.ndarray, parent_pos: np.ndarray,
              generated: bool = False) -> np.ndarray:
        """P(HIV+ | own covariates, referrer state) for a batch.

        ``generated=True`` is used for DDB-generated people; it differs from the
        observed-person mapping only when a ``theta_generated`` variant fitted a
        separate mapping on generated children."""
        feats = theta_features(self.theta_kind, X, parent_X, is_root, parent_pos)
        z = feats @ (self._g_w if generated else self._t_w) + (self._g_b if generated else self._t_b)
        return 1.0 / (1.0 + np.exp(-z))


def theta_features(kind: str, X: np.ndarray, parent_X: np.ndarray, is_root: np.ndarray, parent_pos: np.ndarray) -> np.ndarray:
    cols = [X, is_root[:, None].astype(float), parent_pos[:, None].astype(float)]
    if kind == "parent_features":
        cols.insert(1, parent_X)
    elif kind != "parent_label":
        raise ValueError(f"unknown theta kind {kind}")
    return np.hstack(cols)


def _forest_arrays(forest: Forest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(forest.features, dtype=np.float64)
    y = np.asarray(forest.labels, dtype=np.int64)
    parents = np.asarray(forest.parents, dtype=np.int64)
    return X, y, parents


def fit_theta(forest: Forest, kind: str, seed: int):
    from sklearn.linear_model import LogisticRegression

    X, y, parents = _forest_arrays(forest)
    is_root = parents < 0
    parent_X = np.where(is_root[:, None], 0.0, X[np.maximum(parents, 0)])
    parent_pos = np.where(is_root, 0, y[np.maximum(parents, 0)])
    feats = theta_features(kind, X, parent_X, is_root, parent_pos)
    # Calibrated (unweighted) logistic model: the Gittins index needs probabilities.
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    model.fit(feats, y)
    return model


def fit_theta_generated(ddb: "DDB", forest: Forest, seed: int, reps: int = 10):
    """Variant (not the paper's): theta for generated people, fitted on DDB
    samples of real children (x_child ~ DDB(x_parent)) with the real child's
    label and the real referrer's label, so the oracle's P(+) for generated
    children is calibrated to what DDB actually generates."""
    from sklearn.linear_model import LogisticRegression

    ddb.prepare()
    rng = np.random.default_rng(seed + 1000)
    X, y, parents = _forest_arrays(forest)
    child = np.flatnonzero(parents >= 0)
    px = np.repeat(X[parents[child]], reps, axis=0)
    gx = ddb.sample_child_features(px, rng)
    ppos = np.repeat(y[parents[child]], reps)
    feats = theta_features(ddb.theta_kind, gx, px, np.zeros(len(gx), dtype=bool), ppos)
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    model.fit(feats, np.repeat(y[child], reps))
    return model


def train_ddb(forest: Forest, cat_sizes: Sequence[int], seed: int = 11, theta_kind: str = "parent_features") -> DDB:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X, y, parents = _forest_arrays(forest)
    n, D = X.shape
    counts = np.zeros(n, dtype=float)
    for p in parents:
        if p >= 0:
            counts[p] += 1

    # (1) GPR over |N(v)| (neighbours = successors + referrer) given covariates.
    degree = counts + (parents >= 0)
    kernel = 1.0 * RBF(length_scale=6.0) + WhiteKernel(noise_level=0.4)
    count_model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True, random_state=seed),
    )
    count_model.fit(X, degree)

    # (2) Conditional denoising diffusion over child covariates given parent covariates.
    child_idx = np.flatnonzero(parents >= 0)
    parent_x = X[parents[child_idx]]
    child_x = X[child_idx] * 2.0 - 1.0
    reps = DIFF_REPS
    t = rng.integers(0, DIFF_STEPS, size=(len(child_idx), reps))
    eps = rng.standard_normal((len(child_idx), reps, D))
    ab = ALPHA_BAR[t][..., None]
    z = np.sqrt(ab) * child_x[:, None, :] + np.sqrt(1.0 - ab) * eps
    inputs = np.concatenate(
        [np.repeat(parent_x[:, None, :], reps, axis=1), z, (t / (DIFF_STEPS - 1))[..., None]], axis=2
    ).reshape(-1, 2 * D + 1)
    targets = eps.reshape(-1, D)
    diffusion_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(96, 96), activation="relu", max_iter=220, random_state=seed, early_stopping=True),
    )
    diffusion_model.fit(inputs, targets)

    # (3) Oracle feature mapping theta.
    theta_model = fit_theta(forest, theta_kind, seed)
    return DDB(count_model, diffusion_model, theta_model, D, int(counts.max()), theta_kind, cat_sizes)


# --------------------------------------------------------------------------- oracle


class Grid:
    def __init__(self, gamma: float, size: int):
        self.gamma = gamma
        self.upper = 1.0 / (1.0 - gamma)
        self.m = np.linspace(0.0, self.upper, int(size))
        self.dx = float(self.m[1] - self.m[0])

    def collection(self, prod: np.ndarray) -> np.ndarray:
        """Whittle: Phi(m) = M - int_m^M prod_i dphi_i/dy dy (rows = collections)."""
        trap = 0.5 * self.dx * (prod[:, :-1] + prod[:, 1:])
        suffix = np.zeros_like(prod)
        suffix[:, :-1] = np.cumsum(trap[:, ::-1], axis=1)[:, ::-1]
        return self.upper - suffix

    def derivative(self, phi: np.ndarray) -> np.ndarray:
        return np.clip(np.gradient(phi, self.dx, axis=-1), 0.0, 1.0)

    def index(self, cont: np.ndarray) -> np.ndarray:
        """Retirement value where continuing stops being better (cont - m decreasing)."""
        diff = cont - self.m
        neg = diff < 0
        first = np.argmax(neg, axis=1)
        out = np.full(len(cont), self.upper)
        has = neg.any(axis=1)
        zero = has & (first == 0)
        out[zero] = 0.0
        mid = has & (first > 0)
        i = first[mid]
        rows = np.flatnonzero(mid)
        a = diff[rows, i - 1]
        b = diff[rows, i]
        out[mid] = self.m[i - 1] + self.dx * a / (a - b)
        return out


def _children_product(child_deriv: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Product of child derivative curves per parent (rows of child_deriv grouped by parent)."""
    G = child_deriv.shape[-1]
    prod = np.ones((len(counts), G))
    has = counts > 0
    if has.any():
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])[has]
        prod[has] = np.multiply.reduceat(child_deriv, starts, axis=0)
    return prod


def gittins_indices(levels: List[dict], p_root: np.ndarray, grid: Grid) -> np.ndarray:
    """Gittins index of every level-0 node of a batch of expansions.

    ``levels[L]`` for L >= 1 holds ``p`` (n_L x 2: P(+) given parent label 0/1)
    and ``counts`` of children for level L-1 nodes is ``levels[L]['parent_counts']``.
    Level-0 nodes use ``p_root`` (their actual referrer state is known).
    """
    gamma = grid.gamma
    deriv_below: Optional[np.ndarray] = None  # (n_{L+1}, 2, G): dphi of level L+1 nodes given their parent's label
    for L in range(len(levels) - 1, 0, -1):
        p = levels[L]["p"]
        n = p.shape[0]
        if L + 1 < len(levels):
            counts = levels[L + 1]["parent_counts"]
            phi_c = [grid.collection(_children_product(deriv_below[:, lab, :], counts)) for lab in (0, 1)]
        else:
            phi_c = [np.broadcast_to(grid.m, (n, grid.m.size)) for _ in (0, 1)]
        deriv = np.empty((n, 2, grid.m.size))
        for parent_lab in (0, 1):
            q = p[:, parent_lab][:, None]
            cont = (1.0 - q) * (gamma * phi_c[0]) + q * (1.0 + gamma * phi_c[1])
            deriv[:, parent_lab, :] = grid.derivative(np.maximum(grid.m, cont))
        deriv_below = deriv
    n0 = len(p_root)
    if len(levels) > 1:
        counts = levels[1]["parent_counts"]
        phi_c = [grid.collection(_children_product(deriv_below[:, lab, :], counts)) for lab in (0, 1)]
    else:
        phi_c = [np.broadcast_to(grid.m, (n0, grid.m.size)) for _ in (0, 1)]
    q = p_root[:, None]
    cont = (1.0 - q) * (gamma * phi_c[0]) + q * (1.0 + gamma * phi_c[1])
    return grid.index(cont)


# --------------------------------------------------------------------------- PEGE policy


class PEGEPolicy:
    def __init__(self, ddb: DDB, config: dict):
        self.ddb = ddb
        self.ddb.prepare()
        self.k = int(config["k"])
        self.depth = int(config["depth"])
        self.grid_size = int(config["grid"])
        self.aggregate = config["aggregate"]
        self.seed = int(config["seed"])
        self.oracle_gamma = config.get("oracle_gamma")
        self.rng = np.random.default_rng(self.seed)
        self.grid: Optional[Grid] = None

    def reset(self, state: EpisodeState) -> None:
        self.rng = np.random.default_rng([self.seed, int(state.policy_seed)])
        gamma = float(self.oracle_gamma) if self.oracle_gamma else state.gamma
        self.grid = Grid(gamma, self.grid_size)

    def frontier_inputs(self, state: EpisodeState, frontier: Sequence[int]):
        D = self.ddb.feature_dim
        X = np.asarray([state.features[v] for v in frontier], dtype=np.float64)
        parent_X = np.zeros((len(frontier), D))
        is_root = np.zeros(len(frontier), dtype=bool)
        parent_pos = np.zeros(len(frontier), dtype=np.int64)
        for i, v in enumerate(frontier):
            par = state.parent.get(v)
            if par is None:
                is_root[i] = True
            else:
                parent_X[i] = state.features[par]
                parent_pos[i] = int(state.labels.get(par, 0))
        return X, parent_X, is_root, parent_pos

    def expand(self, X0: np.ndarray, is_root: np.ndarray) -> List[dict]:
        """k independent DDB expansions of depth d from every frontier person."""
        ddb, rng, k = self.ddb, self.rng, self.k
        levels: List[dict] = [{"X": np.repeat(X0, k, axis=0)}]
        if self.depth <= 0:
            return levels
        mean0, std0 = ddb.count_mean_std(X0)
        mean, std = np.repeat(mean0, k), np.repeat(std0, k)
        has_ref = np.repeat(~is_root, k)
        for L in range(1, self.depth + 1):
            prev = levels[L - 1]["X"]
            if L > 1:
                mean, std = ddb.count_mean_std(prev)
                has_ref = np.ones(len(prev), dtype=bool)
            counts = ddb.sample_successors(mean, std, has_ref, rng) if len(prev) else np.zeros(0, dtype=np.int64)
            parent_rows = np.repeat(np.arange(len(prev)), counts)
            parent_X = prev[parent_rows]
            X = ddb.sample_child_features(parent_X, rng)
            n = len(X)
            p = np.empty((n, 2))
            for lab in (0, 1):
                p[:, lab] = ddb.theta(X, parent_X, np.zeros(n, dtype=bool), np.full(n, lab), generated=True)
            levels.append({"X": X, "p": p, "parent_counts": counts})
        return levels

    def choose(self, state: EpisodeState) -> int:
        frontier = list(state.frontier)
        if len(frontier) == 1:
            return frontier[0]
        X0, pX0, is_root, ppos = self.frontier_inputs(state, frontier)
        p0 = self.ddb.theta(X0, pX0, is_root, ppos)
        F, k = len(frontier), self.k
        levels = self.expand(X0, is_root)
        idx = gittins_indices(levels, np.repeat(p0, k), self.grid).reshape(F, k)
        if self.aggregate == "mean":
            score = idx.mean(axis=1)
        elif self.aggregate == "mode":
            winners = np.bincount(idx.argmax(axis=0), minlength=F).astype(float)
            score = winners + 1e-6 * idx.mean(axis=1) / self.grid.upper
        elif self.aggregate == "mean_plus_var":
            score = idx.mean(axis=1) + idx.var(axis=1)
        elif self.aggregate == "mean_minus_var":
            score = idx.mean(axis=1) - idx.var(axis=1)
        else:
            raise ValueError(f"unknown aggregation {self.aggregate}")
        return frontier[int(np.argmax(score))]


# --------------------------------------------------------------------------- CLI


def write_policy_dir(out: Path, ddb_path: Path, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    cfg["ddb"] = str(ddb_path)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "policy.json").write_text(
        json.dumps({"command": ["python", "method/pege_ddb.py", "serve", "--model-dir", str(out)]}, indent=2) + "\n"
    )


def cmd_train(args: argparse.Namespace) -> None:
    out = Path(args.out)
    config = {**DEFAULT_CONFIG, **{k: v for k, v in vars(args).items() if k in DEFAULT_CONFIG and v is not None}}
    forest = load_train_forest(args.data)
    cat_sizes = json.loads(Path(args.data).read_text())["categorical_sizes"]
    ddb = train_ddb(forest, cat_sizes, seed=int(config["seed"]), theta_kind=config["theta"])
    if config.get("theta_generated"):
        ddb.theta_gen_model = fit_theta_generated(ddb, forest, int(config["seed"]))
        ddb._prepared = False
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ddb.pkl").open("wb") as f:
        pickle.dump(ddb, f)
    write_policy_dir(out, out / "ddb.pkl", config)
    print(json.dumps({"trained": str(out), "config": config, "max_children": ddb.max_children}))


def cmd_variant(args: argparse.Namespace) -> None:
    """Same trained DDB, different PEGE settings (optionally a refit theta)."""
    base = Path(args.base)
    out = Path(args.out)
    config = json.loads((base / "config.json").read_text())
    config.pop("ddb", None)
    for key in DEFAULT_CONFIG:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    out.mkdir(parents=True, exist_ok=True)
    with (base / "ddb.pkl").open("rb") as f:
        ddb = pickle.load(f)
    if config["theta"] != ddb.theta_kind:
        ddb.theta_model = fit_theta(load_train_forest(args.data), config["theta"], int(config["seed"]))
        ddb.theta_kind = config["theta"]
        ddb._prepared = False
    if config.get("theta_generated"):
        ddb.theta_gen_model = fit_theta_generated(ddb, load_train_forest(args.data), int(config["seed"]))
        ddb._prepared = False
    with (out / "ddb.pkl").open("wb") as f:
        pickle.dump(ddb, f)
    write_policy_dir(out, out / "ddb.pkl", config)
    print(json.dumps({"variant": str(out), "config": config}))


def cmd_serve(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    config = {**DEFAULT_CONFIG, **json.loads((model_dir / "config.json").read_text())}
    with Path(config.get("ddb", model_dir / "ddb.pkl")).open("rb") as f:
        ddb = pickle.load(f)
    policy = PEGEPolicy(ddb, config)
    serve(policy.choose, policy.reset)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("train", "variant"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--data", default=str(ROOT / "data" / "train_forest.json"))
        p.add_argument("--k", type=int)
        p.add_argument("--depth", type=int)
        p.add_argument("--grid", type=int)
        p.add_argument("--aggregate", choices=["mean", "mode", "mean_plus_var", "mean_minus_var"])
        p.add_argument("--seed", type=int)
        p.add_argument("--theta", choices=["parent_label", "parent_features"])
        p.add_argument("--theta-generated", dest="theta_generated", action="store_const", const=True)
        p.add_argument("--oracle-gamma", dest="oracle_gamma", type=float)
        if name == "variant":
            p.add_argument("--base", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    {"train": cmd_train, "variant": cmd_variant, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    # Dispatch through the importable module so pickled DDB objects reference
    # ``method.pege_ddb.DDB`` (loadable from any entry point), not ``__main__``.
    import importlib

    importlib.import_module("method.pege_ddb").main()
