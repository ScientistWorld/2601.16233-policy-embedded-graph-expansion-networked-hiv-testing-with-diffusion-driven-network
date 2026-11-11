"""Paper baselines (Section 6.2) as SAPOG policies served over the protocol.

* random             -- uniformly random discovered, untested person.
* greedy_neighbor    -- random person whose referrer tested positive, else random.
* greedy_classifier  -- linear SVC with balanced class weights trained on the
                        training forest's covariates; test the frontier person
                        with the highest decision value (network-agnostic).

    python baseline/policies.py train --out-root checkpoints
    python baseline/policies.py serve --kind greedy_classifier --model-dir checkpoints/greedy_classifier
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sapog_data import load_train_forest  # noqa: E402
from data.sapog_protocol import EpisodeState, serve  # noqa: E402

KINDS = ("random", "greedy_neighbor", "greedy_classifier")


def write_manifest(out: Path, kind: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "baseline/policies.py", "serve", "--kind", kind, "--model-dir", str(out)]
    (out / "policy.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")


def cmd_train(args: argparse.Namespace) -> None:
    root = Path(args.out_root)
    for kind in args.kinds.split(","):
        out = root / kind
        write_manifest(out, kind)
        if kind == "greedy_classifier":
            import numpy as np
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC

            forest = load_train_forest(args.data)
            model = make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=20000, random_state=13))
            model.fit(np.asarray(forest.features, dtype=float), np.asarray(forest.labels))
            with (out / "model.pkl").open("wb") as f:
                pickle.dump(model, f)
        print(json.dumps({"baseline": kind, "dir": str(out)}))


def cmd_serve(args: argparse.Namespace) -> None:
    kind = args.kind
    rng = random.Random(0)
    scorer = None
    if kind == "greedy_classifier":
        import numpy as np

        with (Path(args.model_dir) / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        cache: dict = {}

        def scorer(state: EpisodeState, v: int) -> float:
            if v not in cache:
                cache[v] = float(model.decision_function(np.asarray([state.features[v]], dtype=float))[0])
            return cache[v]

    def on_reset(state: EpisodeState) -> None:
        nonlocal rng
        rng = random.Random(state.policy_seed)

    def choose(state: EpisodeState) -> int:
        frontier = list(state.frontier)
        if kind == "random":
            return rng.choice(frontier)
        if kind == "greedy_neighbor":
            hot = [v for v in frontier if state.parent_label(v) == 1]
            return rng.choice(hot or frontier)
        return max(frontier, key=lambda v: scorer(state, v))

    serve(choose, on_reset)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out-root", default=str(ROOT / "checkpoints"))
    t.add_argument("--data", default=str(ROOT / "data" / "train_forest.json"))
    t.add_argument("--kinds", default=",".join(KINDS))
    s = sub.add_parser("serve")
    s.add_argument("--kind", choices=KINDS, required=True)
    s.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    {"train": cmd_train, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()
