"""Summarize hardening runs: paired per-episode differences vs the reference.

Reads notes/hardening/details/*_test.json (per-episode metrics written by the
hidden evaluator's --details flag) and prints / writes a summary with paired
mean differences, their standard errors, the seed spread of the reference and
an illustrative normalized rank (sum of |coefficient| * improvement / std
across the attempts listed here).
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


def load_details():
    rows = {}
    for path in sorted((OUT / "details").glob("*_test.json")):
        raw = json.loads(path.read_text())
        tag = path.name[: -len("_test.json")]
        for name, eps in raw["per_episode"].items():
            if any(e is None for e in eps):
                continue
            key = name if tag == "test_reference" else (tag if tag != "probes" else f"probe:{name}")
            rows[key] = eps
        for name, status in raw.get("status", {}).items():
            if status != "ok":
                key = name if tag == "test_reference" else (tag if tag != "probes" else f"probe:{name}")
                rows.setdefault(key, None)
    return rows


def coefficients():
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())
    exp = ref["experiments"]["synthetic_main"]["metrics"]
    return {m: abs(float(exp[m]["coefficient"])) for m in METRICS if m in exp}


def main() -> None:
    rows = load_details()
    if REF not in rows or rows[REF] is None:
        print("reference details missing")
        return
    ref = rows[REF]
    n = len(ref)
    means = {}
    for name, eps in rows.items():
        if eps is None:
            means[name] = {m: 0.0 for m in METRICS}
        else:
            means[name] = {m: st.mean(e[m] for e in eps) for m in METRICS}
    paired = {}
    for name, eps in rows.items():
        if eps is None or name == REF:
            continue
        paired[name] = {}
        for m in METRICS:
            d = [a[m] - b[m] for a, b in zip(eps, ref)]
            sd = st.stdev(d) if len(d) > 1 else 0.0
            paired[name][m] = {"diff": st.mean(d), "se": sd / math.sqrt(len(d))}
    cohort = [k for k in rows if not k.startswith("probe:")]
    coef = coefficients()
    spread = {m: st.pstdev([means[k][m] for k in cohort]) or 1.0 for m in METRICS}
    rank = {k: sum(coef.get(m, 0.0) * (means[k][m] - means[REF][m]) / spread[m] for m in METRICS) for k in rows}
    seeds = [k for k in (REF, "seed12", "seed13") if k in means]
    seed_spread = {m: {"min": min(means[k][m] for k in seeds), "max": max(means[k][m] for k in seeds),
                       "range": max(means[k][m] for k in seeds) - min(means[k][m] for k in seeds)} for m in METRICS}
    summary = {"episodes": n, "means": means, "paired_vs_reference": paired, "normalized_rank": rank,
               "rank_cohort": cohort, "rank_std": spread, "seed_spread": seed_spread, "seeds": seeds}
    fog = OUT / "scores" / "fog_validation.json"
    if fog.exists():
        summary["fog_validation"] = json.loads(fog.read_text())["fully_observable_gittins_validation"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"episodes={n}")
    print(f"{'method':28s} " + " ".join(f"{m:>12s}" for m in METRICS) + f" {'dAUC':>8s} {'se':>6s} {'rank':>8s}")
    for name in sorted(rows, key=lambda k: -means[k]["auc"]):
        p = paired.get(name, {}).get("auc", {"diff": 0.0, "se": 0.0})
        print(f"{name:28s} " + " ".join(f"{means[name][m]:12.4f}" for m in METRICS)
              + f" {p['diff']:8.3f} {p['se']:6.3f} {rank[name]:8.3f}")
    print("seed spread:", json.dumps(seed_spread))
    if "fog_validation" in summary:
        print("FOG (validation, privileged):", json.dumps(summary["fog_validation"]))


if __name__ == "__main__":
    main()
