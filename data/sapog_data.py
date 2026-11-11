"""Loaders for the public SAPOG HIV-referral data (stdlib only).

``data/train_forest.json``        training forest: referral trees with covariates
                                  and HIV labels, plus five component-level folds.
``data/validation_episodes.json`` visible validation episodes, each a forest of
                                  ~155 people (the same size and distribution as
                                  the held-out test episodes).

Features are stored as 72-character bit strings (17 categorical answers,
one-hot encoded).  ``parents[i] == -1`` marks a root (a recruited seed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DATA_DIR = Path(__file__).resolve().parent
TRAIN_FILE = DATA_DIR / "train_forest.json"
VALIDATION_FILE = DATA_DIR / "validation_episodes.json"


def _bits(rows: Sequence[str]) -> List[List[int]]:
    return [[1 if ch == "1" else 0 for ch in row] for row in rows]


@dataclass
class Forest:
    features: List[List[int]]
    labels: List[int]
    parents: List[int]
    roots: List[int]
    components: List[int] = field(default_factory=list)
    folds: List[List[int]] = field(default_factory=list)
    children: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.children:
            self.children = {i: [] for i in range(len(self.parents))}
            for i, p in enumerate(self.parents):
                if p >= 0:
                    self.children[p].append(i)

    def __len__(self) -> int:
        return len(self.labels)


def load_train_forest(path: str | Path = TRAIN_FILE) -> Forest:
    raw = json.loads(Path(path).read_text())
    return Forest(
        features=_bits(raw["features"]),
        labels=[int(v) for v in raw["labels"]],
        parents=[int(v) for v in raw["parents"]],
        roots=[int(v) for v in raw["roots"]],
        components=[int(v) for v in raw["components"]],
        folds=[[int(c) for c in fold] for fold in raw["folds"]],
    )


def load_validation_episodes(path: str | Path = VALIDATION_FILE) -> List[Tuple[int, Forest]]:
    raw = json.loads(Path(path).read_text())
    out = []
    for item in raw["episodes"]:
        out.append(
            (
                int(item["episode_seed"]),
                Forest(
                    features=_bits(item["features"]),
                    labels=[int(v) for v in item["labels"]],
                    parents=[int(v) for v in item["parents"]],
                    roots=[int(v) for v in item["roots"]],
                ),
            )
        )
    return out


def fold_split(forest: Forest, fold: int) -> Tuple[List[int], List[int]]:
    """Component-level split of the training forest (train nodes, held-out nodes)."""
    held = set(forest.folds[fold])
    train = [i for i, c in enumerate(forest.components) if c not in held]
    test = [i for i, c in enumerate(forest.components) if c in held]
    return train, test
