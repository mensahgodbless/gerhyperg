"""
Render confusion matrices for trained runs into Markdown.

Usage:
    python scripts/aggregate_confusion.py
    python scripts/aggregate_confusion.py --ablations main holistic
    python scripts/aggregate_confusion.py --output paper/tables/confusion.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

CLASS_NAMES = ("positive", "neutral", "negative")

# Where the confusion_matrix lives inside metrics.json and what to glob.
DATASET_SCHEMAS = {
    "vgaf": {
        "label":         "VGAF (sum across 3 seeds, official Val 766 clips per seed)",
        "child_glob":    "seed_*/metrics.json",
        "metric_subkey": "test_metrics",     # confusion lives at metrics["test_metrics"]["confusion_matrix"]
    },
    "gecv": {
        "label":         "GECV (sum across 10 folds, total 408 clips)",
        "child_glob":    "fold_*/metrics.json",
        "metric_subkey": "val_metrics",
    },
}


def render_matrix(name: str, cm: list[list[int]]) -> str:
    """Render a 3x3 matrix as a Markdown table + per-class recall/precision."""
    if len(cm) != 3 or any(len(row) != 3 for row in cm):
        return f"### {name}\n\n(matrix has unexpected shape, skipping)\n"

    # Recall = TP / row_sum, precision = TP / col_sum.
    row_sums = [sum(row) for row in cm]
    col_sums = [sum(cm[r][c] for r in range(3)) for c in range(3)]
    total = sum(row_sums)
    diag = sum(cm[i][i] for i in range(3))
    overall_acc = diag / total if total else float("nan")

    lines = [f"### {name}", ""]
    lines.append("**Confusion matrix** (rows = true, cols = predicted):")
    lines.append("")
    lines.append("|        | " + " | ".join(f"pred {c}" for c in CLASS_NAMES) + " | row total |")
    lines.append("|--------|" + "|".join("---" for _ in CLASS_NAMES) + "|---|")
    for i, name_ in enumerate(CLASS_NAMES):
        cells = " | ".join(f"{cm[i][j]:d}" for j in range(3))
        lines.append(f"| true {name_} | {cells} | {row_sums[i]} |")
    lines.append("| **col total** | "
                 + " | ".join(f"**{cs}**" for cs in col_sums)
                 + f" | **{total}** |")
    lines.append("")

    # Per-class recall and precision.
    lines.append("**Per-class metrics:**")
    lines.append("")
    lines.append("| Class | Recall | Precision | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for i, name_ in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        recall = tp / row_sums[i] if row_sums[i] else float("nan")
        precision = tp / col_sums[i] if col_sums[i] else float("nan")
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0
        lines.append(f"| {name_} | {recall:.4f} | {precision:.4f} | {f1:.4f} | {row_sums[i]} |")
    lines.append("")
    lines.append(f"**Overall accuracy:** {overall_acc:.4f} ({diag}/{total})")
    lines.append("")
    return "\n".join(lines)


def sum_matrices(matrices: list[list[list[int]]]) -> Optional[list[list[int]]]:
    """Element-wise sum of a list of 3x3 matrices."""
    if not matrices:
        return None
    out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for cm in matrices:
        for i in range(3):
            for j in range(3):
                out[i][j] += int(cm[i][j])
    return out


def collect_matrices(
    run_dir: Path,
    schema: dict,
) -> list[list[list[int]]]:
    """Gather all confusion matrices from a run dir's children."""
    out = []
    for child in sorted(run_dir.glob(schema["child_glob"])):
        try:
            m = json.loads(child.read_text())
        except json.JSONDecodeError:
            continue
        # Confusion matrix lives one level deep.
        inner = m.get(schema["metric_subkey"], {})
        cm = inner.get("confusion_matrix")
        if cm is None:
            continue
        out.append(cm)
    return out


def render_dataset(
    dataset: str,
    results_root: Path,
    schema: dict,
    ablation_filter: Optional[set[str]] = None,
) -> Optional[str]:
    runs = []
    for d in sorted(results_root.glob(f"{dataset}_*")):
        ablation = d.name.replace(f"{dataset}_", "")
        if ablation_filter is not None and ablation not in ablation_filter:
            continue
        matrices = collect_matrices(d, schema)
        if not matrices:
            continue
        summed = sum_matrices(matrices)
        runs.append((ablation, summed, len(matrices)))

    if not runs:
        return None

    lines = [f"## {schema['label']}", ""]
    for ablation, summed, n_children in runs:
        title = f"{ablation} (aggregated over {n_children} children)"
        lines.append(render_matrix(title, summed))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SCHEMAS),
                        help="Which datasets to include (default: all known).")
    parser.add_argument("--ablations", nargs="*", default=None,
                        help="Restrict to specific ablations (default: all found).")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("CONFUSION_MATRICES.md"))
    args = parser.parse_args()

    ablation_filter = set(args.ablations) if args.ablations else None
    blocks = ["# Confusion matrices", ""]
    if ablation_filter:
        blocks.append(f"Restricted to ablations: {', '.join(sorted(ablation_filter))}")
        blocks.append("")

    for dataset in args.datasets:
        if dataset not in DATASET_SCHEMAS:
            print(f"WARN: unknown dataset {dataset!r}, skipping.")
            continue
        rendered = render_dataset(
            dataset, args.results_root, DATASET_SCHEMAS[dataset], ablation_filter,
        )
        if rendered is None:
            blocks.append(f"## {DATASET_SCHEMAS[dataset]['label']}\n\n(no runs found)\n")
        else:
            blocks.append(rendered)

    md = "\n".join(blocks).rstrip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(md)
    print(f"Wrote {args.output}.")


if __name__ == "__main__":
    main()
