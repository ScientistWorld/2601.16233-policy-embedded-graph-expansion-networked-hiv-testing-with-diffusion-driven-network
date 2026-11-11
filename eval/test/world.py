"""Hidden generative world for the SAPOG HIV-referral benchmark (stdlib only).

This module is the single source of every forest in the benchmark.  The
public training forest and the visible validation episodes under ``data/`` were
written once by ``python eval/test/world.py --write-public`` from the public
seeds below; held-out test episodes are regenerated at test time from the
hidden seeds in ``eval/test/evaluate.py`` and never touch disk.

Each person carries two latent traits -- infection risk ``r`` and referral
activity ``s`` -- that are inherited (noisily) along referral edges.  A
positive parent raises the child's risk, socially active parents refer more
contacts and those contacts are riskier.  Observable covariates are 17
categorical answers (one-hot, 72 dims) whose category probabilities depend on
the latent traits with item-specific loadings plus parent-to-child homophily.
Labels depend only on latent risk, so covariates are informative but noisy.

Everything uses ``random.Random`` primitives built on ``rng.random()`` so the
output is identical across Python versions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

CAT_SIZES: List[int] = [2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6]
FEATURE_DIM = sum(CAT_SIZES)  # 72

# (risk loading, activity loading, homophily) for each categorical item.
ITEM_ROLES = [
    (0.0, 0.0, -1.0),  # sex: partners mostly opposite (special-cased)
    (1.3, 0.0, 0.15),
    (0.0, 1.3, 0.15),
    (0.9, 0.4, 0.25),
    (0.0, 0.0, 0.60),
    (1.1, 0.0, 0.25),
    (0.0, 1.2, 0.20),
    (0.6, 0.7, 0.30),
    (0.0, 0.0, 0.50),
    (0.3, 0.0, 0.40),
    (1.0, 0.0, 0.20),
    (0.0, 1.0, 0.30),
    (0.0, 0.0, 0.50),
    (0.7, 0.3, 0.20),
    (0.2, 0.9, 0.30),
    (0.5, 0.5, 0.40),
    (0.0, 0.0, 0.60),
]

DESIGN_SEED = 20260117
ITEM_SHARPNESS = 1.30

# Latent dynamics.
ROOT_RISK_MEAN = 0.40
RISK_INHERIT = 0.45
ACTIVITY_TO_CHILD_RISK = 0.75
POSITIVE_PARENT_RISK = 0.70
RISK_NOISE = 0.75
ACTIVITY_INHERIT = 0.40
LABEL_INTERCEPT = -5.00
LABEL_SLOPE = 1.55
BRANCH_INTERCEPT = -1.10
BRANCH_SLOPE = 0.85
MAX_CHILDREN = 7
MAX_COMPONENT = 45
MAX_DEPTH = 9

EPISODE_NODES = 155
TRAIN_NODES = 778
PUBLIC_TRAIN_SEED = 51001
PUBLIC_VALIDATION_SEEDS = list(range(61001, 61021))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normal(rng: random.Random) -> float:
    u1 = max(rng.random(), 1e-300)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _poisson(rng: random.Random, lam: float) -> int:
    limit = math.exp(-lam)
    k = 0
    prod = rng.random()
    while prod > limit:
        k += 1
        prod *= rng.random()
    return k


def _categorical(rng: random.Random, logits: Sequence[float]) -> int:
    top = max(logits)
    weights = [math.exp(v - top) for v in logits]
    u = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if u <= acc:
            return i
    return len(weights) - 1


def _design() -> Dict[str, list]:
    rng = random.Random(DESIGN_SEED)
    risk_w, act_w, base = [], [], []
    for size in CAT_SIZES:
        levels = [(-1.0 + 2.0 * i / (size - 1)) for i in range(size)]
        r_levels = levels[:]
        a_levels = levels[:]
        for i in range(size - 1, 0, -1):
            j = int(rng.random() * (i + 1))
            r_levels[i], r_levels[j] = r_levels[j], r_levels[i]
        for i in range(size - 1, 0, -1):
            j = int(rng.random() * (i + 1))
            a_levels[i], a_levels[j] = a_levels[j], a_levels[i]
        risk_w.append(r_levels)
        act_w.append(a_levels)
        base.append([0.6 * _normal(rng) for _ in range(size)])
    return {"risk": risk_w, "activity": act_w, "base": base}


_DESIGN = _design()


def _sample_items(rng: random.Random, r: float, s: float, parent_items: List[int] | None) -> List[int]:
    items: List[int] = []
    for j, size in enumerate(CAT_SIZES):
        a_r, a_s, homophily = ITEM_ROLES[j]
        if j == 0:
            if parent_items is None:
                items.append(0 if rng.random() < 0.5 else 1)
            else:
                items.append(1 - parent_items[0] if rng.random() < 0.8 else parent_items[0])
            continue
        if parent_items is not None and rng.random() < homophily:
            items.append(parent_items[j])
            continue
        logits = [
            ITEM_SHARPNESS * (a_r * r * _DESIGN["risk"][j][k] + a_s * s * _DESIGN["activity"][j][k]) + _DESIGN["base"][j][k]
            for k in range(size)
        ]
        items.append(_categorical(rng, logits))
    return items


def _one_hot(items: List[int]) -> List[int]:
    out: List[int] = []
    for j, size in enumerate(CAT_SIZES):
        block = [0] * size
        block[items[j]] = 1
        out.extend(block)
    return out


def generate_component(rng: random.Random, max_size: int = MAX_COMPONENT) -> List[dict]:
    r = ROOT_RISK_MEAN + _normal(rng)
    s = _normal(rng)
    items = _sample_items(rng, r, s, None)
    label = 1 if rng.random() < _sigmoid(LABEL_INTERCEPT + LABEL_SLOPE * r) else 0
    nodes = [{"parent": -1, "items": items, "label": label, "r": r, "s": s, "depth": 0}]
    queue = [0]
    head = 0
    while head < len(queue):
        idx = queue[head]
        head += 1
        node = nodes[idx]
        if node["depth"] >= MAX_DEPTH:
            continue
        lam = math.exp(BRANCH_INTERCEPT + BRANCH_SLOPE * node["s"])
        # Seeds (roots) are recruited because they refer at least one contact.
        n_child = _poisson(rng, lam) + (1 if node["depth"] == 0 else 0)
        n_child = min(MAX_CHILDREN, n_child, max_size - len(nodes))
        for _ in range(max(0, n_child)):
            s_c = ACTIVITY_INHERIT * node["s"] + math.sqrt(1.0 - ACTIVITY_INHERIT ** 2) * _normal(rng)
            r_c = (
                RISK_INHERIT * node["r"]
                + ACTIVITY_TO_CHILD_RISK * node["s"]
                + POSITIVE_PARENT_RISK * node["label"]
                + RISK_NOISE * _normal(rng)
            )
            items_c = _sample_items(rng, r_c, s_c, node["items"])
            label_c = 1 if rng.random() < _sigmoid(LABEL_INTERCEPT + LABEL_SLOPE * r_c) else 0
            nodes.append({"parent": idx, "items": items_c, "label": label_c, "r": r_c, "s": s_c, "depth": node["depth"] + 1})
            queue.append(len(nodes) - 1)
    return nodes


def generate_forest(seed: int, target_nodes: int, keep_latent: bool = False) -> dict:
    """Generate components until ``target_nodes`` people are reached."""
    rng = random.Random(seed)
    features: List[List[int]] = []
    labels: List[int] = []
    parents: List[int] = []
    components: List[int] = []
    roots: List[int] = []
    latent: List[List[float]] = []
    comp = 0
    while len(labels) < target_nodes:
        remaining = target_nodes - len(labels)
        nodes = generate_component(rng, max_size=min(MAX_COMPONENT, max(1, remaining)))
        offset = len(labels)
        roots.append(offset)
        for node in nodes:
            features.append(_one_hot(node["items"]))
            labels.append(int(node["label"]))
            parents.append(-1 if node["parent"] < 0 else offset + node["parent"])
            components.append(comp)
            latent.append([node["r"], node["s"]])
        comp += 1
    out = {
        "features": features,
        "labels": labels,
        "parents": parents,
        "components": components,
        "roots": roots,
    }
    if keep_latent:
        out["latent"] = latent
    return out


def episode(seed: int, keep_latent: bool = False) -> dict:
    return generate_forest(seed, EPISODE_NODES, keep_latent=keep_latent)


def _bits(rows: List[List[int]]) -> List[str]:
    return ["".join("1" if v else "0" for v in row) for row in rows]


def _component_folds(components: List[int], n_folds: int = 5) -> List[List[int]]:
    sizes: Dict[int, int] = {}
    for c in components:
        sizes[c] = sizes.get(c, 0) + 1
    bins: List[List[int]] = [[] for _ in range(n_folds)]
    load = [0] * n_folds
    counts = [0] * n_folds
    for c in sorted(sizes, key=lambda c: (-sizes[c], c)):
        j = min(range(n_folds), key=lambda i: (counts[i], load[i]))
        bins[j].append(c)
        load[j] += sizes[c]
        counts[j] += 1
    return bins


def write_public(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = generate_forest(PUBLIC_TRAIN_SEED, TRAIN_NODES)
    train_payload = {
        "description": "Public training forest for the SAPOG HIV-referral benchmark. "
        "Each component is a referral tree; features are 17 one-hot categorical answers (72 bits).",
        "feature_dim": FEATURE_DIM,
        "categorical_sizes": CAT_SIZES,
        "features": _bits(train["features"]),
        "labels": train["labels"],
        "parents": train["parents"],
        "components": train["components"],
        "roots": train["roots"],
        "folds": _component_folds(train["components"]),
    }
    (out_dir / "train_forest.json").write_text(json.dumps(train_payload, separators=(",", ":")) + "\n")
    episodes = []
    for seed in PUBLIC_VALIDATION_SEEDS:
        ep = episode(seed)
        episodes.append(
            {
                "episode_seed": seed,
                "features": _bits(ep["features"]),
                "labels": ep["labels"],
                "parents": ep["parents"],
                "roots": ep["roots"],
            }
        )
    val_payload = {
        "description": "Visible validation episodes (same distribution as the hidden test episodes).",
        "feature_dim": FEATURE_DIM,
        "categorical_sizes": CAT_SIZES,
        "episodes": episodes,
    }
    (out_dir / "validation_episodes.json").write_text(json.dumps(val_payload, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-public", metavar="DIR", help="write data/train_forest.json and data/validation_episodes.json")
    parser.add_argument("--stats", type=int, default=0, help="print summary stats over N fresh episodes")
    args = parser.parse_args()
    if args.write_public:
        write_public(Path(args.write_public))
        print(json.dumps({"wrote": args.write_public}))
    if args.stats:
        n = pos = comps = 0
        child_counts: List[int] = []
        sizes: List[int] = []
        for i in range(args.stats):
            ep = generate_forest(900000 + i, EPISODE_NODES)
            n += len(ep["labels"])
            pos += sum(ep["labels"])
            comps += len(ep["roots"])
            cc = [0] * len(ep["labels"])
            for p in ep["parents"]:
                if p >= 0:
                    cc[p] += 1
            child_counts.extend(cc)
            for c in set(ep["components"]):
                sizes.append(ep["components"].count(c))
        print(json.dumps({
            "nodes_per_episode": n / args.stats,
            "positive_rate": pos / n,
            "components_per_episode": comps / args.stats,
            "mean_children": sum(child_counts) / len(child_counts),
            "max_children": max(child_counts),
            "max_component": max(sizes),
            "size_hist": {k: sizes.count(k) for k in sorted(set(sizes))},
        }, indent=1))


if __name__ == "__main__":
    main()
