"""Visible train-slice scorer for SAPOG testing policies (stdlib only).

Every subdirectory ``checkpoints/<name>/`` is a submitted method.  Its
``policy.json`` names a command; the scorer starts that command once per
validation episode and plays the episode interactively over JSON lines
(protocol documented in ``briefing/evaluation.md`` and implemented for
policies by ``data/sapog_protocol.py``).  The policy only ever sees the root
people, the people revealed by its own tests, and the labels of the people it
tested.

Validation episodes come from ``data/validation_episodes.json`` (visible, same
distribution and size as the held-out episodes).  Gamma, the budget (test
every discovered person, ``T = |V|``), the metrics and the aggregation are
fixed here.  Any failure (missing/invalid ``policy.json``, crash, timeout, an
action that is not a currently discovered untested person) scores the method
with the worst value (0.0) in every metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SLICE = "train"
EXPERIMENT = "synthetic_main"
GAMMA = 0.99
METRICS = ("auc", "reward_25pct", "reward_50pct", "reward_75pct")
CHECKPOINTS = (0.25, 0.50, 0.75)
WORST = {m: 0.0 for m in METRICS}  # rewards are non-negative, so 0 is the floor

VALIDATION_FILE = ROOT / "data" / "validation_episodes.json"

STEP_TIMEOUT_S = float(os.environ.get("SAPOG_STEP_TIMEOUT", "900"))
MAX_IGNORED_LINES = 10_000


# ---------------------------------------------------------------- episodes


def _policy_seed(seed: int) -> int:
    digest = hashlib.sha256(f"sapog-policy-{seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def load_episodes() -> List[dict]:
    raw = json.loads(VALIDATION_FILE.read_text())
    episodes = []
    for item in raw["episodes"]:
        seed = int(item["episode_seed"])
        features = [[1 if ch == "1" else 0 for ch in bits] for bits in item["features"]]
        labels = [int(v) for v in item["labels"]]
        parents = [int(v) for v in item["parents"]]
        n = len(labels)
        children: List[List[int]] = [[] for _ in range(n)]
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        rng = random.Random(f"ids-{seed}")
        ids = rng.sample(range(100_000, 1_000_000), n)
        roots = [int(r) for r in item["roots"]]
        rng.shuffle(roots)
        for kids in children:
            rng.shuffle(kids)
        episodes.append(
            {
                "policy_seed": _policy_seed(seed),
                "features": features,
                "labels": labels,
                "parents": parents,
                "children": children,
                "roots": roots,
                "ids": ids,
            }
        )
    return episodes


# ---------------------------------------------------------------- metrics


def discounted_auc(curve: List[float]) -> float:
    if not curve:
        return 0.0
    if len(curve) == 1:
        return float(curve[0])
    area = sum((curve[i] + curve[i + 1]) * 0.5 for i in range(len(curve) - 1))
    return area / len(curve)


def episode_metrics(labels_in_order: List[int], budget: int) -> Dict[str, float]:
    cumulative: List[float] = []
    total = 0.0
    for t in range(budget):
        y = labels_in_order[t] if t < len(labels_in_order) else 0
        total += (GAMMA ** t) * y
        cumulative.append(total)
    out = {"auc": discounted_auc(cumulative)}
    for frac, key in zip(CHECKPOINTS, METRICS[1:]):
        idx = max(0, min(budget - 1, int(round(budget * frac)) - 1))
        out[key] = cumulative[idx]
    return out


def aggregate(rows: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total = float(sum(w for w, _ in rows))
    out = {}
    for m in METRICS:
        value = sum(w * r[m] for w, r in rows) / total
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"impossible aggregate value for {m}: {value}")
        out[m] = value
    return out


# ---------------------------------------------------------------- policy process


class PolicyFailure(Exception):
    pass


def _node_msg(ep: dict, idx: int) -> dict:
    parent = ep["parents"][idx]
    return {
        "id": ep["ids"][idx],
        "parent": None if parent < 0 else ep["ids"][parent],
        "features": ep["features"][idx],
    }


def _reader(stream, out: "queue.Queue[Optional[str]]") -> None:
    try:
        for line in stream:
            out.put(line)
    except Exception:
        pass
    out.put(None)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def run_episode(command: List[str], ep: dict, log_path: Path) -> Dict[str, float]:
    n = len(ep["labels"])
    budget = n
    id_to_idx = {nid: i for i, nid in enumerate(ep["ids"])}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:  # command not found, etc.
            raise PolicyFailure(f"could not start policy: {exc}") from exc
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(target=_reader, args=(proc.stdout, lines), daemon=True).start()

        def send(msg: dict) -> None:
            try:
                proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
                proc.stdin.flush()
            except Exception as exc:
                raise PolicyFailure(f"policy stdin closed: {exc}") from exc

        def receive_action() -> int:
            ignored = 0
            deadline = time.monotonic() + STEP_TIMEOUT_S
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PolicyFailure("policy timed out")
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    raise PolicyFailure("policy timed out")
                if line is None:
                    raise PolicyFailure(f"policy exited early (code {proc.poll()})")
                try:
                    msg = json.loads(line)
                except ValueError:
                    msg = None
                if isinstance(msg, dict) and "action" in msg:
                    action = msg["action"]
                    if isinstance(action, bool) or not isinstance(action, int):
                        raise PolicyFailure(f"non-integer action {action!r}")
                    return action
                ignored += 1
                if ignored > MAX_IGNORED_LINES:
                    raise PolicyFailure("too many non-protocol output lines")

        try:
            frontier = set(ep["roots"])
            tested: List[int] = []
            send(
                {
                    "type": "reset",
                    "policy_seed": ep["policy_seed"],
                    "gamma": GAMMA,
                    "budget": budget,
                    "feature_dim": len(ep["features"][0]),
                    "new_nodes": [_node_msg(ep, i) for i in ep["roots"]],
                }
            )
            for t in range(budget):
                action_id = receive_action()
                idx = id_to_idx.get(action_id)
                if idx is None or idx not in frontier:
                    raise PolicyFailure(f"step {t}: action {action_id} is not a discovered untested person")
                frontier.discard(idx)
                tested.append(idx)
                revealed = ep["children"][idx]
                frontier.update(revealed)
                if t + 1 < budget:
                    send(
                        {
                            "type": "step",
                            "t": t + 1,
                            "tested": action_id,
                            "label": int(ep["labels"][idx]),
                            "new_nodes": [_node_msg(ep, c) for c in revealed],
                        }
                    )
            try:
                send({"type": "end"})
            except PolicyFailure:
                pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            _kill(proc)
    if len(tested) != budget or len(set(tested)) != budget:
        raise PolicyFailure("episode did not test every person exactly once")
    return episode_metrics([int(ep["labels"][i]) for i in tested], budget)


# ---------------------------------------------------------------- methods


def discover_methods(methods_root: Path) -> List[Path]:
    if not methods_root.is_dir():
        return []
    return sorted(p for p in methods_root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def load_command(method_dir: Path) -> List[str]:
    manifest = method_dir / "policy.json"
    if not manifest.is_file():
        raise PolicyFailure("missing policy.json")
    try:
        raw = json.loads(manifest.read_text())
    except Exception as exc:
        raise PolicyFailure(f"unreadable policy.json: {exc}") from exc
    command = raw.get("command") if isinstance(raw, dict) else None
    if not isinstance(command, list) or not command or not all(isinstance(c, str) and c for c in command):
        raise PolicyFailure("policy.json must contain a non-empty 'command' list of strings")
    return command


def write_scores(path: Path, results: Dict[str, Dict[str, float]], status: Dict[str, str]) -> None:
    payload = {
        "slice": SLICE,
        "experiments": {EXPERIMENT: {"slice": SLICE, "results": results}},
        "_status": status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def default_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    return max(1, int(os.environ.get("SAPOG_EVAL_WORKERS", n)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods-root", default=str(ROOT / "checkpoints"))
    parser.add_argument("--output", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--details", default=None, help="optional per-episode metrics JSON (analysis only)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    output = Path(args.output)
    methods = discover_methods(Path(args.methods_root))
    names = [m.name for m in methods]
    results: Dict[str, Dict[str, float]] = {name: dict(WORST) for name in names}
    status: Dict[str, str] = {name: "failed: evaluation did not finish" for name in names}
    write_scores(output, results, status)
    if not methods:
        print(json.dumps({"error": f"no method directories under {args.methods_root}"}))
        return 1

    episodes = load_episodes()
    log_root = ROOT / "logs" / "eval_train"
    commands: Dict[str, List[str]] = {}
    for method in methods:
        try:
            commands[method.name] = load_command(method)
        except PolicyFailure as exc:
            status[method.name] = f"failed: {exc}"

    per_episode: Dict[str, List[Optional[Dict[str, float]]]] = {n: [None] * len(episodes) for n in commands}
    failures: Dict[str, str] = {}
    lock = threading.Lock()

    def job(name: str, e: int) -> None:
        with lock:
            if name in failures:
                return
        try:
            row = run_episode(commands[name], episodes[e], log_root / name / f"episode{e:03d}.log")
        except PolicyFailure as exc:
            with lock:
                failures.setdefault(name, f"episode {e}: {exc}")
            return
        except Exception as exc:  # evaluator-side problem for this policy: still a failure
            with lock:
                failures.setdefault(name, f"episode {e}: {type(exc).__name__}: {exc}")
            return
        with lock:
            per_episode[name][e] = row

    workers = args.workers or default_workers()
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, name, e) for name in commands for e in range(len(episodes))]
        for f in futures:
            f.result()

    for name in commands:
        if name in failures:
            status[name] = f"failed: {failures[name]}"
            continue
        rows = per_episode[name]
        if any(r is None for r in rows):
            status[name] = "failed: missing episodes"
            continue
        try:
            results[name] = aggregate([(len(ep["labels"]), r) for ep, r in zip(episodes, rows)])
            status[name] = "ok"
        except ValueError as exc:
            status[name] = f"failed: {exc}"
    write_scores(output, results, status)
    if args.details:
        Path(args.details).parent.mkdir(parents=True, exist_ok=True)
        Path(args.details).write_text(json.dumps({"episodes": len(episodes), "per_episode": per_episode, "status": status}) + "\n")
    print(json.dumps({"wrote": str(output), "slice": SLICE, "status": status, "seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
