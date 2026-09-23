"""
Aggregate per-run summary.json files into a single Markdown table.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --datasets vgaf
    python scripts/aggregate_results.py --output paper/tables/main_results.md
    python scripts/aggregate_results.py --results_root some/other/dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

# Each dataset has a different summary schema. Keys to look for, in order:
#   - mean_key, std_key  : the headline accuracy mean and its std
#   - f1_mean_key        : weighted-F1 mean
#   - f1_std_key         : weighted-F1 std
#   - n_key              : something to count (seeds, folds); optional
#   - per_key            : per-seed/fold list for the extra column
DATASET_SCHEMAS = {
    "vgaf": {
        "label":       "VGAF (test on official Val)",
        "mean_key":    "test_acc_mean",
        "std_key":     "test_acc_std",
        "f1_mean_key": "test_f1_mean",
        "f1_std_key":  "test_f1_std",
        "n_key":       "seeds",        # list, so n = len(seeds)
        "per_key":     "per_seed_acc",
        "per_label":   "Per-seed acc",
    },
    "gecv": {
        "label":       "GECV (10-fold CV)",
        "mean_key":    "val_acc_mean",
        "std_key":     "val_acc_std",
        "f1_mean_key": "val_f1_mean",
        "f1_std_key":  "val_f1_std",
        "n_key":       "folds_run",    # list of fold indices
        "per_key":     "per_fold_acc",
        "per_label":   "Per-fold acc",
    },
}


def render_dataset(
    dataset: str,
    results_root: Path,
    schema: dict,
) -> Optional[str]:
    """Render one dataset's table. Returns None if no runs found."""
    runs = []
    for d in sorted(results_root.glob(f"{dataset}_*")):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            s = json.loads(summary_path.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: skipping unreadable {summary_path}: {e}")
            continue
        # Schema check — be permissive but warn.
        if schema["mean_key"] not in s:
            print(f"WARN: {summary_path} has no '{schema['mean_key']}', skipping.")
            continue
        ablation = d.name.replace(f"{dataset}_", "")
        runs.append((ablation, s))

    if not runs:
        return None

    lines = [f"## {schema['label']}", ""]
    header = f"| Ablation | Acc | F1 | Acc std | n | {schema['per_label']} |"
    sep = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for ablation, s in runs:
        acc = s[schema["mean_key"]]
        std = s.get(schema["std_key"], 0.0)
        f1 = s.get(schema["f1_mean_key"], float("nan"))
        n_field = s.get(schema["n_key"])
        n = len(n_field) if isinstance(n_field, list) else (n_field or "")
        per_field = s.get(schema["per_key"], [])
        per = " / ".join(f"{x:.4f}" for x in per_field) if per_field else "—"
        lines.append(
            f"| {ablation} | {acc:.4f} | {f1:.4f} | ±{std:.4f} | {n} | {per} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SCHEMAS),
                        help="Which datasets to include (default: all known).")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("RESULTS_TABLE.md"))
    args = parser.parse_args()

    blocks = ["# All experimental results", ""]
    total_rows = 0
    for dataset in args.datasets:
        if dataset not in DATASET_SCHEMAS:
            print(f"WARN: unknown dataset {dataset!r}, skipping.")
            continue
        rendered = render_dataset(dataset, args.results_root, DATASET_SCHEMAS[dataset])
        if rendered is None:
            blocks.append(f"## {DATASET_SCHEMAS[dataset]['label']}\n\n(no runs found)\n")
        else:
            blocks.append(rendered)
            total_rows += rendered.count("\n| ") - 1  # subtract header
        blocks.append("")

    md = "\n".join(blocks).rstrip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(md)
    print(f"Wrote {args.output} ({total_rows} run rows).")


if __name__ == "__main__":
    main()
