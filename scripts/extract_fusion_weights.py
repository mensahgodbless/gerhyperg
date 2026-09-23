"""Extract learned decision-fusion weights (alpha_m) from trained checkpoints.

Reads `fusion.weight_logits` from every best_model.pt under --results_root,
applies softmax to recover the normalised fusion weights alpha_m, and
aggregates mean +/- std across runs (5 seeds for VGAF, 10 folds for GECV)
per configuration.

Single-modality configurations are silently skipped (their fusion module
either does not exist or has num_branches=1 with a trivial softmax of 1.0).

This script does NOT load the model. It only inspects the state-dict, so
it runs in seconds against the full sweep.

MODALITY ORDERING ASSUMPTION:
    The 1D `fusion.weight_logits` tensor is interpreted positionally as
        3-branch:  [visual, audio, scene]
        4-branch:  [visual, audio, scene, temporal]
    matching Eq. 10 in the paper. VERIFY this matches the order in which
    branch_logits are passed to DecisionFusion.forward() in your model
    (look for the line that constructs the tuple, e.g.
    `self.fusion((visual_logits, audio_logits, scene_logits, ...))`).
    If your model passes them in a different order, edit the constants
    MODALITY_ORDER_3BR / MODALITY_ORDER_4BR below.

Usage:
    python scripts/extract_fusion_weights.py
    python scripts/extract_fusion_weights.py --results_root results/
    python scripts/extract_fusion_weights.py --output results/fusion_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


MODALITY_ORDER_3BR = ["visual", "audio", "scene"]
MODALITY_ORDER_4BR = ["visual", "audio", "scene", "temporal"]
FUSION_KEY = "fusion.weight_logits"



def _unwrap_state_dict(ckpt):
    """Some training loops save {'state_dict': ..., 'optimizer': ...};
    others save the state_dict directly. Handle both."""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt


def _extract_one(ckpt_path: Path):
    """Returns (raw_logits, softmax_weights) as 1D numpy arrays, or None
    if the checkpoint has no multi-branch fusion (single-modality config)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _unwrap_state_dict(ckpt)
    if FUSION_KEY not in sd:
        return None
    logits = sd[FUSION_KEY].detach().cpu().float()
    if logits.ndim != 1 or logits.numel() < 2:
        # num_branches=1 (single-modality wrapped fusion) or per_class layout: skip.
        return None
    weights = torch.softmax(logits, dim=0)
    return logits.numpy(), weights.numpy()




def _discover(results_root: Path) -> dict[str, list[Path]]:
    """For each immediate subdirectory of results_root, find run-dirs that
    contain best_model.pt. Returns {config_name: [run_dir, ...]}, sorted."""
    out: dict[str, list[Path]] = defaultdict(list)
    if not results_root.is_dir():
        raise FileNotFoundError(f"results_root not found: {results_root}")
    for config_dir in sorted(results_root.iterdir()):
        if not config_dir.is_dir():
            continue
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "best_model.pt").is_file():
                out[config_dir.name].append(run_dir)
    return dict(out)




def _aggregate_config(run_dirs: list[Path]) -> dict | None:
    """Walks the run-dirs for one configuration, extracts fusion weights from
    each, and returns aggregated statistics. Returns None if no multi-branch
    fusion was found in any run (single-modality config)."""
    per_run = []
    for rd in run_dirs:
        res = _extract_one(rd / "best_model.pt")
        if res is None:
            continue
        logits, weights = res
        per_run.append({
            "run": rd.name,
            "weight_logits": logits.tolist(),
            "alpha": weights.tolist(),
        })
    if not per_run:
        return None

    alphas = np.array([r["alpha"] for r in per_run])  # (n_runs, n_branches)
    n_branches = alphas.shape[1]
    if n_branches == 3:
        modalities = MODALITY_ORDER_3BR
    elif n_branches == 4:
        modalities = MODALITY_ORDER_4BR
    else:
        # Unexpected: not 3 or 4 branches. Fall back to positional names.
        modalities = [f"branch_{i}" for i in range(n_branches)]

    means = alphas.mean(axis=0)
    stds  = alphas.std(axis=0)  # ddof=0, matches paper convention

    return {
        "n_runs": len(per_run),
        "n_branches": n_branches,
        "modalities": modalities,
        "alpha_mean": dict(zip(modalities, means.tolist())),
        "alpha_std":  dict(zip(modalities, stds.tolist())),
        "alpha_min":  dict(zip(modalities, alphas.min(axis=0).tolist())),
        "alpha_max":  dict(zip(modalities, alphas.max(axis=0).tolist())),
        "per_run": per_run,
    }




def _print_table(summary: dict):
    """Pretty per-config table grouped by n_branches."""
    for nb in (3, 4):
        configs = [c for c in sorted(summary)
                   if summary[c]["n_branches"] == nb]
        if not configs:
            continue
        modalities = MODALITY_ORDER_3BR if nb == 3 else MODALITY_ORDER_4BR
        equal_weight = 1.0 / nb
        print(f"\n=== {nb}-branch configurations  (equal-weight reference = {equal_weight:.3f}) ===")
        # Header
        header = f"  {'config':28s}  {'n':>3s}  "
        header += "  ".join(f"alpha_{m:9s}" for m in modalities)
        print(header)
        # Rows
        for cfg in configs:
            s = summary[cfg]
            cells = []
            for m in modalities:
                mu = s["alpha_mean"][m]
                sd = s["alpha_std"][m]
                cells.append(f"{mu:.3f}\u00b1{sd:.3f}")
            row = f"  {cfg:28s}  {s['n_runs']:>3d}  " + "  ".join(f"{c:15s}" for c in cells)
            print(row)


def _print_paper_summary(summary: dict):

    print("\n=== Paper paste-in summary ===")
    print("For each (dataset x branch-count), alphas are pooled across configs.\n")

    for dataset_tag, dataset_label in [("vgaf_", "VGAF"), ("gecv_", "GECV")]:
        for nb in (3, 4):
            equal_weight = 1.0 / nb
            modalities = MODALITY_ORDER_3BR if nb == 3 else MODALITY_ORDER_4BR
            configs = [c for c in sorted(summary)
                       if c.startswith(dataset_tag)
                       and summary[c]["n_branches"] == nb
                       and not c.endswith("_only")]
            if not configs:
                continue
            print(f"{dataset_label} {nb}-branch  "
                  f"(n_configs={len(configs)}, equal-weight={equal_weight:.3f}):")
            # Pool: collect each config's alpha_mean per modality, then average
            # across configs. This is mean-of-means, not pooled-by-run, but for
            # a paper paste-in line it is what we want.
            pooled = {m: [] for m in modalities}
            for c in configs:
                for m in modalities:
                    pooled[m].append(summary[c]["alpha_mean"][m])
            means_across_configs = {m: float(np.mean(pooled[m])) for m in modalities}
            mins  = {m: float(np.min(pooled[m]))  for m in modalities}
            maxs  = {m: float(np.max(pooled[m]))  for m in modalities}
            for m in modalities:
                deviation_from_uniform = (means_across_configs[m] - equal_weight) * 100
                print(f"  alpha_{m:9s}  mean={means_across_configs[m]:.3f}  "
                      f"range=[{mins[m]:.3f}, {maxs[m]:.3f}]  "
                      f"({deviation_from_uniform:+.1f}pp from uniform)")
            smallest = min(means_across_configs, key=means_across_configs.get)
            largest  = max(means_across_configs, key=means_across_configs.get)
            spread   = means_across_configs[largest] - means_across_configs[smallest]
            print(f"  --> smallest mean alpha: {smallest} ({means_across_configs[smallest]:.3f}); "
                  f"largest: {largest} ({means_across_configs[largest]:.3f}); "
                  f"spread: {spread*100:.1f}pp\n")




def main():
    parser = argparse.ArgumentParser(
        description="Extract learned fusion weights from trained checkpoints.")
    parser.add_argument("--results_root", type=Path, default=Path("results"),
                        help="Parent dir containing per-config subdirectories.")
    parser.add_argument("--output", type=Path,
                        default=Path("results/fusion_weights.json"),
                        help="Where to write the per-run + aggregated JSON.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-config table; only print paper summary.")
    args = parser.parse_args()

    print(f"Scanning {args.results_root} for best_model.pt ...")
    runs = _discover(args.results_root)
    if not runs:
        sys.exit(f"No config-dirs with best_model.pt found under {args.results_root}")
    print(f"Found {len(runs)} config-directories.\n")

    summary = {}
    skipped = []
    for config_name in sorted(runs):
        agg = _aggregate_config(runs[config_name])
        if agg is None:
            skipped.append(config_name)
            continue
        summary[config_name] = agg

    if skipped:
        print(f"Skipped {len(skipped)} single-modality config(s) "
              f"(no multi-branch fusion):")
        for c in skipped:
            print(f"  - {c}")

    if not args.quiet:
        _print_table(summary)

    _print_paper_summary(summary)

    # Sanity check: warn if audio is the LARGEST mean alpha anywhere
    for cfg, s in summary.items():
        means = s["alpha_mean"]
        if "audio" in means and means["audio"] == max(means.values()):
            print(f"WARNING: in {cfg}, alpha_audio={means['audio']:.3f} is the "
                  f"LARGEST modality weight. Either training is unusual for "
                  f"this config, or your MODALITY_ORDER_* constants do not "
                  f"match the order in which branch_logits are passed to "
                  f"DecisionFusion.forward(). Verify against your model code.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()