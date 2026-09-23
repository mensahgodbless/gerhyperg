"""
Aggregate per-class accuracies for VGAF (5 seeds) and GECV (10 folds).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Iterable

# Configuration ordering for the final table (matches paper presentation).
ABLATIONS_ORDER = [
    "visual_only",
    "audio_only",
    "scene_only",
    "temporal_only",
    "main",            # hypergraph 3-branch
    "no_pose",
    "mean_pool",
    "gat",
    "holistic",
    "hypergraph_temporal",
    "no_pose_temporal",
    "mean_pool_temporal",
    "gat_temporal",
    "holistic_temporal",
]

# Display name overrides for the markdown / LaTeX output.
DISPLAY_NAME = {
    "main": "Hypergraph (3-br)",
    "no_pose": "No-pose (3-br)",
    "mean_pool": "Mean-pool (3-br)",
    "gat": "GAT (3-br)",
    "holistic": "Holistic (3-br)",
    "hypergraph_temporal": "Hypergraph (4-br)",
    "no_pose_temporal": "No-pose (4-br)",
    "mean_pool_temporal": "Mean-pool (4-br)",
    "gat_temporal": "GAT (4-br)",
    "holistic_temporal": "Holistic (4-br)",
    "visual_only": "Visual only",
    "audio_only": "Audio only",
    "scene_only": "Scene only",
    "temporal_only": "Temporal only",
}


def collect_runs(dataset_root: Path) -> list[float]:
    """Return all per-run metrics.json paths under a given results/{dataset}_{ablation}/
    directory, searching seed_*/ and fold_*/ subdirs.
    """
    if not dataset_root.exists():
        return []
    subdirs = list(dataset_root.glob("seed_*/")) + list(dataset_root.glob("fold_*/"))
    return sorted([d / "metrics.json" for d in subdirs if (d / "metrics.json").exists()])


def load_metrics(metrics_path: Path) -> dict | None:
    """Return the metrics dict from a single metrics.json, or None if missing.

    VGAF uses 'test_metrics' (per-seed test on official val split).
    GECV uses 'val_metrics' (per-fold val in 10-fold CV).
    Try both.
    """
    try:
        with metrics_path.open() as f:
            blob = json.load(f)
        return blob.get("test_metrics") or blob.get("val_metrics")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [warn] could not parse {metrics_path}: {exc}")
        return None


def stats(vals: Iterable[float]) -> tuple[float, float, int]:
    vals = list(vals)
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def aggregate_one(ablation_dir: Path) -> dict | None:
    """Aggregate test_metrics across all runs (seeds or folds) in one ablation dir."""
    run_paths = collect_runs(ablation_dir)
    if not run_paths:
        return None

    overall_acc, pos_acc, neu_acc, neg_acc = [], [], [], []
    n_loaded = 0
    for p in run_paths:
        m = load_metrics(p)
        if m is None:
            continue
        overall_acc.append(m["accuracy"])
        pc = m.get("per_class_accuracy")
        if pc is None or len(pc) != 3:
            print(f"  [warn] missing/malformed per_class_accuracy in {p}")
            continue
        pos_acc.append(pc[0])
        neu_acc.append(pc[1])
        neg_acc.append(pc[2])
        n_loaded += 1

    return {
        "n_runs": n_loaded,
        "overall": stats(overall_acc),
        "positive": stats(pos_acc),
        "neutral": stats(neu_acc),
        "negative": stats(neg_acc),
    }


def fmt_cell(stat: tuple[float, float, int]) -> str:
    mean, sd, n = stat
    if n == 0:
        return "—"
    return f"{100*mean:.2f} ± {100*sd:.2f}"


def render_markdown(per_dataset: dict[str, dict[str, dict]]) -> str:
    """Render a markdown table with both datasets side by side."""
    lines = []
    header = (
        "| Configuration | n | VGAF Overall | VGAF Pos | VGAF Neu | VGAF Neg "
        "| n | GECV Overall | GECV Pos | GECV Neu | GECV Neg |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines.append(header)
    lines.append(sep)

    for abl in ABLATIONS_ORDER:
        vrow = per_dataset.get("vgaf", {}).get(abl)
        grow = per_dataset.get("gecv", {}).get(abl)
        if vrow is None and grow is None:
            continue
        display = DISPLAY_NAME.get(abl, abl)

        def cells_for(row):
            if row is None:
                return ["—"] * 5
            return [
                str(row["n_runs"]),
                fmt_cell(row["overall"]),
                fmt_cell(row["positive"]),
                fmt_cell(row["neutral"]),
                fmt_cell(row["negative"]),
            ]

        cells = [display] + cells_for(vrow) + cells_for(grow)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to the results/ root containing {dataset}_{ablation}/ subdirs",
    )
    parser.add_argument("--out-md", default="per_class_table.md")
    args = parser.parse_args()

    root = Path(args.results_dir)
    per_dataset: dict[str, dict[str, dict]] = {"vgaf": {}, "gecv": {}}

    for dataset in ("vgaf", "gecv"):
        for abl in ABLATIONS_ORDER:
            abl_dir = root / f"{dataset}_{abl}"
            agg = aggregate_one(abl_dir)
            if agg is not None and agg["overall"][2] > 0:
                per_dataset[dataset][abl] = agg

    # Sanity-check: warn if any expected configuration is missing.
    for dataset in ("vgaf", "gecv"):
        missing = [a for a in ABLATIONS_ORDER if a not in per_dataset[dataset]]
        if missing:
            print(f"[warn] {dataset}: missing aggregates for {missing}")

    md = render_markdown(per_dataset)

    print(md)

    Path(args.out_md).write_text(md)



if __name__ == "__main__":
    main()