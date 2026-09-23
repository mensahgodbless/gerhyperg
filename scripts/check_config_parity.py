"""
Verify a new ablation would train under the same config as a published run.
Exit code 0 = parity, 1 = unexpected differences found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for c in [here] + list(here.parents):
        if (c / "configs").is_dir() and (c / "requirements.txt").is_file():
            return c
    raise FileNotFoundError("project root not found")


_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.train_utils import load_config  # noqa: E402  (unchanged original)

# Keys the decomposition ablations are meant to change, plus run plumbing.
EXPECTED_DIFFS = {
    "model.visual.backbone",
    "model.visual.temporal_pool",
    "project.device",
}


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _parse_overrides(items):
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        try:
            cast: object = int(val)
        except ValueError:
            try:
                cast = float(val)
            except ValueError:
                cast = {"true": True, "false": False}.get(val.lower(), val)
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = cast
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", type=Path, required=True,
                    help="config.yaml saved inside a published run's seed/fold dir")
    ap.add_argument("--dataset", default="vgaf")
    ap.add_argument("--ablation", action="append", default=[])
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.published) as f:
        pub = _flatten(yaml.safe_load(f))
    new = _flatten(load_config(args.dataset, args.ablation,
                               _parse_overrides(args.override)))

    # "attn" for the comparison.
    pub.setdefault("model.visual.temporal_pool", "attn")
    new.setdefault("model.visual.temporal_pool", "attn")

    unexpected, expected = [], []
    for k in sorted(set(pub) | set(new)):
        a, b = pub.get(k, "<absent>"), new.get(k, "<absent>")
        if a != b:
            (expected if k in EXPECTED_DIFFS else unexpected).append((k, a, b))

    print(f"published: {args.published}")
    print(f"new:       {args.dataset} + {args.ablation} + {args.override}\n")
    if expected:
        print("Expected differences (ablation keys):")
        for k, a, b in expected:
            print(f"  {k:40s} {a!r:>14} -> {b!r}")
        print()
    if unexpected:
        print("UNEXPECTED differences. New runs would NOT be comparable:")
        for k, a, b in unexpected:
            print(f"  {k:40s} {a!r:>14} -> {b!r}")
        return 1
    print("Parity OK: only the ablation keys differ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
