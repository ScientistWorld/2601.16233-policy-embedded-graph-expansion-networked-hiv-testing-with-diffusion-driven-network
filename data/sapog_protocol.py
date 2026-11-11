"""Policy side of the SAPOG evaluation protocol (stdlib only).

A method is a directory ``checkpoints/<name>/`` containing ``policy.json``::

    {"command": ["python", "method/my_policy.py", "--model-dir", "checkpoints/<name>"]}

The evaluators run that command from the workspace root (with the workspace on
``PYTHONPATH``) once per episode and exchange one JSON object per line:

    evaluator -> policy  {"type": "reset", "policy_seed": int, "gamma": 0.99,
                          "budget": T, "feature_dim": 72,
                          "new_nodes": [{"id": int, "parent": null, "features": [0/1]*72}, ...]}
    policy -> evaluator  {"action": <id of a discovered, untested person>}
    evaluator -> policy  {"type": "step", "t": int, "tested": id, "label": 0|1,
                          "new_nodes": [{"id": int, "parent": id, "features": [...]}, ...]}
    policy -> evaluator  {"action": id}
    ...
    evaluator -> policy  {"type": "end"}

``serve`` below handles the bookkeeping: write a function that receives the
current :class:`EpisodeState` and returns the id to test next.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional


class EpisodeState:
    """Everything revealed so far in one episode."""

    def __init__(self, reset: dict) -> None:
        self.policy_seed: int = int(reset.get("policy_seed", 0))
        self.gamma: float = float(reset.get("gamma", 0.99))
        self.budget: int = int(reset.get("budget", 0))
        self.feature_dim: int = int(reset.get("feature_dim", 0))
        self.t: int = 0
        self.features: Dict[int, List[int]] = {}
        self.parent: Dict[int, Optional[int]] = {}
        self.children: Dict[int, List[int]] = {}
        self.labels: Dict[int, int] = {}
        self.frontier: List[int] = []  # discovered, untested (discovery order)
        self.roots: List[int] = []
        self._add(reset.get("new_nodes", []))

    def _add(self, nodes: List[dict]) -> None:
        for node in nodes:
            nid = int(node["id"])
            par = node.get("parent")
            self.features[nid] = [int(v) for v in node["features"]]
            self.parent[nid] = None if par is None else int(par)
            self.children.setdefault(nid, [])
            if par is None:
                self.roots.append(nid)
            else:
                self.children.setdefault(int(par), []).append(nid)
            self.frontier.append(nid)

    def observe(self, step: dict) -> None:
        tested = int(step["tested"])
        self.labels[tested] = int(step["label"])
        if tested in self.frontier:
            self.frontier.remove(tested)
        self.t = int(step.get("t", self.t + 1))
        self._add(step.get("new_nodes", []))

    def parent_label(self, nid: int) -> Optional[int]:
        """Label of ``nid``'s referrer, or None for roots."""
        par = self.parent.get(nid)
        return None if par is None else self.labels.get(par)


def serve(choose: Callable[[EpisodeState], int], on_reset: Optional[Callable[[EpisodeState], None]] = None) -> None:
    """Run the protocol loop on stdin/stdout until the evaluator sends ``end``."""
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr  # stray prints must not corrupt the protocol stream
    state: Optional[EpisodeState] = None
    pending: Optional[int] = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "end":
            break
        if kind == "reset":
            state = EpisodeState(msg)
            if on_reset is not None:
                on_reset(state)
        elif kind == "step":
            assert state is not None, "step before reset"
            if pending is not None and int(msg["tested"]) != pending:
                raise RuntimeError("evaluator reported a different tested node")
            state.observe(msg)
        else:
            continue
        pending = int(choose(state))
        out.write(json.dumps({"action": pending}) + "\n")
        out.flush()
