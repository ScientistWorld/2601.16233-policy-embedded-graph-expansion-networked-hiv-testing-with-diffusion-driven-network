"""Check that a fresh end-to-end reproduction gives the recorded reference scores."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ok = True
for slice_ in ("train", "test"):
    new = json.loads((ROOT / "scoring" / f"scores_{slice_}.json").read_text())["experiments"]["synthetic_main"]["results"]
    old = json.loads((ROOT / "notes" / "hardening" / "scores" / f"official_{slice_}.json").read_text())["experiments"]["synthetic_main"]["results"]
    for name, metrics in old.items():
        diff = max(abs(metrics[m] - new.get(name, {}).get(m, float("nan"))) for m in metrics)
        same = diff < 1e-9
        ok &= same
        print(f"{slice_:5s} {name:18s} max|diff|={diff:.3g} {'identical' if same else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
