"""Summarize hardening runs: paired per-episode differences vs the reference.

Reads the per-episode metrics written by the hidden evaluator's --details flag
under notes/hardening/details/:

* test_reference.json  reference method + baselines (official run)
* rules_test.json      hand-written lookahead rules (probes)
* probes_test.json     shortcut probes (prefixed ``probe:``)
* <variant>_test.json  one honest variant each

and prints / writes notes/hardening/summary.json with mean metrics, paired mean
differences vs the reference with standard errors, the reference's seed
spread, an illustrative normalized rank (sum over metrics of |coefficient| *
improvement / std across the non-probe attempts listed here) and the
before/after comparison with round 1 (previous DDB) on the 80 episodes both
rounds share.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notes" / "hardening"
METRICS = ["auc", "reward_25pct", "reward_50pct", "reward_75pct"]
REF = "pege_ddb"
MULTI = {"test_reference": "", "rules": "", "probes": "probe:"}


def load_details(folder: Path):
    rows, status = {}, {}
    for path in sorted(folder.glob("*.json")):
        stem = path.stem
        tag = stem[: -len("_test")] if stem.endswith("_test") else stem
        if tag.startswith("train") or tag == "test_reference_final":
            continue
        raw = json.loads(path.read_text())
        for name, eps in raw["per_episode"].items():
            key = (MULTI[tag] + name) if tag in MULTI else tag
            ok = raw.get("status", {}).get(name) == "ok" and all(e is not None for e in eps)
            rows[key] = eps if ok else None
            status[key] = raw.get("status", {}).get(name, "?")
        for name, s in raw.get("status", {}).items():
            key = (MULTI[tag] + name) if tag in MULTI else tag
            if s != "ok":
                rows.setdefault(key, None)
                status[key] = s
    return rows, status


def coefficients():
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())
    exp = ref["experiments"]["synthetic_main"]["metrics"]
    return {m: abs(float(exp[m]["coefficient"])) for m in METRICS if m in exp}


def weighted_mean(eps, sizes, m):
    return sum(n * e[m] for n, e in zip(sizes, eps)) / sum(sizes)


def paired(a, b, m):
    d = [x[m] - y[m] for x, y in zip(a, b)]
    return {"diff": st.mean(d), "se": (st.stdev(d) / math.sqrt(len(d))) if len(d) > 1 else 0.0}


def episode_sizes(n):
    cache = OUT / "episode_sizes.json"  # written once; avoids touching eval/test while it is locked
    if cache.exists():
        sizes = json.loads(cache.read_text())
        if len(sizes) >= n:
            return sizes[:n]
    import sys

    sys.path.insert(0, str(ROOT / "eval" / "test"))
    import world  # hidden generator (analysis only)

    base = 7_340_117
    return [len(world.episode(base + 104_729 * i)["labels"]) for i in range(n)]


def main() -> None:
    rows, status = load_details(OUT / "details")
    if rows.get(REF) is None:
        print("reference details missing")
        return
    ref = rows[REF]
    n = len(ref)
    sizes = episode_sizes(n)
    means = {k: ({m: weighted_mean(v, sizes, m) for m in METRICS} if v is not None else {m: 0.0 for m in METRICS})
             for k, v in rows.items()}
    pair = {k: {m: paired(v, ref, m) for m in METRICS} for k, v in rows.items() if v is not None and k != REF}
    cohort = [k for k in rows if not k.startswith("probe:")]
    coef = coefficients()
    spread = {m: (st.pstdev([means[k][m] for k in cohort]) or 1.0) for m in METRICS}
    rank = {k: sum(coef.get(m, 0.0) * (means[k][m] - means[REF][m]) / spread[m] for m in METRICS) for k in rows}
    seeds = [k for k in (REF, "seed12", "seed13") if rows.get(k) is not None]
    seed_spread = {m: max(means[k][m] for k in seeds) - min(means[k][m] for k in seeds) for m in METRICS}
    summary = {"episodes": n, "means": means, "status": status, "paired_vs_reference": pair,
               "normalized_rank": rank, "rank_cohort": cohort, "rank_std": spread,
               "seeds": seeds, "seed_range": seed_spread}

    old_rows, _ = load_details(OUT / "round1" / "details")
    if old_rows.get(REF) is not None:
        k = len(old_rows[REF])
        sz = sizes[:k]
        before = {}
        for name in ("pege_ddb", "greedy_classifier", "greedy_neighbor", "random", "theta_parent_features",
                     "no_expansion", "k2", "k48", "seed12", "seed13", "mode", "mean_plus_var", "mean_minus_var"):
            if old_rows.get(name) is not None:
                before[name] = {m: weighted_mean(old_rows[name], sz, m) for m in METRICS}
        summary["round1_first80"] = before
        summary["new_reference_first80"] = {m: weighted_mean(ref[:k], sz, m) for m in METRICS}
        summary["new_vs_round1_reference_first80"] = {m: paired(ref[:k], old_rows[REF], m) for m in METRICS}

    fog = OUT / "scores" / "fog_validation.json"
    if fog.exists():
        summary["fog_validation"] = json.loads(fog.read_text())["fully_observable_gittins_validation"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"episodes={n}")
    print(f"{'method':32s} " + " ".join(f"{m:>12s}" for m in METRICS) + f" {'dAUC':>8s} {'se':>6s} {'rank':>8s}")
    for name in sorted(rows, key=lambda k: -means[k]["auc"]):
        p = pair.get(name, {}).get("auc", {"diff": 0.0, "se": 0.0})
        print(f"{name:32s} " + " ".join(f"{means[name][m]:12.4f}" for m in METRICS)
              + f" {p['diff']:8.3f} {p['se']:6.3f} {rank[name]:8.3f}  {'' if status.get(name) == 'ok' else status.get(name, '')}")
    print("seed range:", json.dumps(seed_spread))
    if "new_vs_round1_reference_first80" in summary:
        print("new reference vs round-1 reference on the shared 80 episodes:",
              json.dumps(summary["new_vs_round1_reference_first80"]["auc"]))
    if "fog_validation" in summary:
        print("FOG (validation, privileged):", json.dumps(summary["fog_validation"]))


if __name__ == "__main__":
    main()
