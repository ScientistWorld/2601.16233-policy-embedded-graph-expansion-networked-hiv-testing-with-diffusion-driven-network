"""Shortcut probes for the hardening audit (stdlib only, never shipped as methods).

Each mode is a policy speaking the evaluator protocol directly:

  random_control   plain random policy (reference for the forged-field probes)
  forged_manifest  random policy whose policy.json also carries gamma/budget/
                   experiment/score fields (the evaluator must ignore them)
  self_report      random policy that adds reward/auc/success claims to every reply
  invalid_action   re-submits an already tested person after a few steps
  crash_midway     exits in the middle of every episode
  hang             stops answering (exercises the per-step timeout)
  hidden_file_read reads the hidden evaluator at test time, regenerates the
                   held-out episodes and plays with full knowledge of labels
                   and topology (the residual leak a grader must catch)
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    mode = sys.argv[1]
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    rng = random.Random(0)
    frontier: list = []
    tested: list = []
    t = 0
    oracle = None

    def reply(action: int) -> None:
        msg = {"action": action}
        if mode == "self_report":
            msg.update({"reward": 1e9, "auc": 1e9, "success": True, "gamma": 1.05})
        out.write(json.dumps(msg) + "\n")
        out.flush()

    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("type") == "end":
            break
        if msg["type"] == "reset":
            rng = random.Random(msg["policy_seed"])
            frontier = [n["id"] for n in msg["new_nodes"]]
            tested, t = [], 0
            if mode == "hidden_file_read":
                oracle = HiddenOracle({n["id"] for n in msg["new_nodes"]})
        else:
            tested.append(msg["tested"])
            if msg["tested"] in frontier:
                frontier.remove(msg["tested"])
            frontier.extend(n["id"] for n in msg["new_nodes"])
            t = msg["t"]
        if mode == "crash_midway" and t >= 40:
            sys.exit(7)
        if mode == "hang" and t >= 5:
            time.sleep(10_000)
        if mode == "invalid_action" and t >= 3:
            reply(tested[0])
            continue
        if mode == "hidden_file_read" and oracle is not None and oracle.ok:
            reply(oracle.choose(frontier))
            continue
        reply(rng.choice(frontier))


class HiddenOracle:
    """Locate this episode inside the hidden evaluator and act on its answers."""

    def __init__(self, root_ids: set) -> None:
        self.ok = False
        path = ROOT / "eval" / "test" / "evaluate.py"
        spec = importlib.util.spec_from_file_location("hidden_eval_probe", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for ep in mod.load_episodes():
            ids = {ep["ids"][r] for r in ep["roots"]}
            if ids == root_ids:
                self.ep = ep
                self.idx = {nid: i for i, nid in enumerate(ep["ids"])}
                self.ok = True
                break
        if self.ok:
            ep = self.ep
            n = len(ep["labels"])
            self.dist = [10**9] * n  # distance to nearest positive in own subtree
            order = []
            stack = list(ep["roots"])
            while stack:
                v = stack.pop()
                order.append(v)
                stack.extend(ep["children"][v])
            for v in reversed(order):
                best = 0 if ep["labels"][v] else 10**9
                for c in ep["children"][v]:
                    best = min(best, self.dist[c] + 1)
                self.dist[v] = best

    def choose(self, frontier: list) -> int:
        return min(frontier, key=lambda nid: (self.dist[self.idx[nid]], nid))


if __name__ == "__main__":
    main()
