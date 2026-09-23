"""
Aggregate cross-dataset eval reports into a single Markdown table.

Usage:
    python scripts/aggregate_cross_eval.py
    python scripts/aggregate_cross_eval.py --source_dataset vgaf --target_dataset gecv
    python scripts/aggregate_cross_eval.py --output paper/tables/cross_eval.md
    python scripts/aggregate_cross_eval.py --results_root some/other/dir
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Tuple


def mean_std(xs: list[float]) -> Tuple[float, float]:
    """Sample mean and std. Std is 0 for a single point, NaN for an empty list."""
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def discover_reports(
    results_root: Path,
    source_dataset: str,
    target_dataset: str,
) -> dict[str, list[tuple[str, dict]]]:
    """Find all cross-eval reports, indexed by ablation → [(seed, report), ...]."""
    pattern = f"{source_dataset}_*/seed_*/cross_eval_{target_dataset}/report.json"
    by_ablation: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for rp in sorted(results_root.glob(pattern)):
        # results / vgaf_main / seed_42 / cross_eval_gecv / report.json
        ablation = rp.parts[-4].replace(f"{source_dataset}_", "")
        seed = rp.parts[-3].replace("seed_", "")
        try:
            report = json.loads(rp.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: skipping unreadable {rp}: {e}")
            continue
        by_ablation[ablation].append((seed, report))
    return by_ablation


def render(
    by_ablation: dict[str, list[tuple[str, dict]]],
    source_dataset: str,
    target_dataset: str,
) -> str:
    if not by_ablation:
        return (
            f"# Cross-dataset evaluation: {source_dataset} → {target_dataset}\n\n"
            f"No reports found.\n"
        )

    lines = [
        f"# Cross-dataset evaluation: trained on {source_dataset.upper()}, "
        f"tested on {target_dataset.upper()}",
        "",
        f"Each architecture was trained on {source_dataset.upper()} (multi-seed), "
        f"then evaluated on the full {target_dataset.upper()} dataset without any "
        f"{target_dataset.upper()} training. Overall accuracy + F1 are mean ± std "
        f"across seeds. Face-count stratified rows report accuracy and weighted-F1 "
        f"per bucket (mean across seeds).",
        "",
    ]

    # ── Overall ─────────────────────────────────────────────────────
    lines += [
        "## Overall",
        "",
        "| Architecture | Acc (mean ± std) | F1 (mean ± std) | Per-seed acc |",
        "|---|---|---|---|",
    ]
    for ablation in sorted(by_ablation):
        seeds_reports = by_ablation[ablation]
        accs = [r["overall"]["accuracy"] for _, r in seeds_reports]
        f1s = [r["overall"]["weighted_f1"] for _, r in seeds_reports]
        am, asd = mean_std(accs)
        fm, fsd = mean_std(f1s)
        per = " / ".join(f"{a:.4f}" for a in accs)
        lines.append(f"| {ablation} | {am:.4f} ± {asd:.4f} | {fm:.4f} ± {fsd:.4f} | {per} |")

    # ── Discover bucket names from the first report (assumed consistent). ──
    first_report = next(iter(by_ablation.values()))[0][1]
    bucket_names = list(first_report["face_count_stratified"].keys())
    bucket_n = {b: first_report["face_count_stratified"][b]["n"] for b in bucket_names}
    header = "| Architecture | " + " | ".join(f"{b} (n={bucket_n[b]})" for b in bucket_names) + " |"
    sep = "|---|" + "|".join("---" for _ in bucket_names) + "|"

    # ── Stratified accuracy ──────────────────────────────────────────
    lines += ["", "## Face-count stratified accuracy (mean across seeds)", "", header, sep]
    for ablation in sorted(by_ablation):
        row = [ablation]
        for b in bucket_names:
            accs = [
                r["face_count_stratified"][b]["accuracy"]
                for _, r in by_ablation[ablation]
                if r["face_count_stratified"].get(b, {}).get("n", 0) > 0
                and "accuracy" in r["face_count_stratified"].get(b, {})
            ]
            row.append("—" if not accs else f"{mean_std(accs)[0]:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Stratified weighted-F1 ───────────────────────────────────────
    lines += ["", "## Face-count stratified weighted-F1 (mean across seeds)", "", header, sep]
    for ablation in sorted(by_ablation):
        row = [ablation]
        for b in bucket_names:
            f1s = [
                r["face_count_stratified"][b]["weighted_f1"]
                for _, r in by_ablation[ablation]
                if r["face_count_stratified"].get(b, {}).get("n", 0) > 0
                and "weighted_f1" in r["face_count_stratified"].get(b, {})
            ]
            row.append("—" if not f1s else f"{mean_std(f1s)[0]:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Per-seed detail ──────────────────────────────────────────────
    lines += ["", "## Per-seed detail", ""]
    for ablation in sorted(by_ablation):
        lines += [
            f"### {ablation}",
            "",
            "| Seed | Overall acc | Overall F1 | " + " | ".join(bucket_names) + " |",
            "|---|---|---|" + "|".join("---" for _ in bucket_names) + "|",
        ]
        for seed, r in sorted(by_ablation[ablation]):
            bvals = []
            for b in bucket_names:
                ent = r["face_count_stratified"].get(b, {})
                if ent.get("n", 0) > 0 and "accuracy" in ent:
                    bvals.append(f"{ent['accuracy']:.4f}")
                else:
                    bvals.append("—")
            lines.append(
                f"| {seed} | {r['overall']['accuracy']:.4f} | "
                f"{r['overall']['weighted_f1']:.4f} | " + " | ".join(bvals) + " |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dataset", default="vgaf",
                        help="The dataset the models were TRAINED on (default: vgaf).")
    parser.add_argument("--target_dataset", default="gecv",
                        help="The dataset the models were EVALUATED on (default: gecv).")
    parser.add_argument("--results_root", type=Path, default=Path("results"),
                        help="Where to walk for `<source>_*/seed_*/cross_eval_<target>/report.json`.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output Markdown file. Default: CROSS_EVAL_<SOURCE>_TO_<TARGET>.md")
    args = parser.parse_args()

    output_path = args.output or Path(
        f"CROSS_EVAL_{args.source_dataset.upper()}_TO_{args.target_dataset.upper()}.md"
    )

    by_ablation = discover_reports(
        args.results_root, args.source_dataset, args.target_dataset
    )
    md = render(by_ablation, args.source_dataset, args.target_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)

    print(md)
    print()
    n_ablations = len(by_ablation)
    n_reports = sum(len(v) for v in by_ablation.values())
    print(f"Wrote {output_path} ({len(md)} chars; "
          f"{n_ablations} ablations, {n_reports} reports).")


if __name__ == "__main__":
    main()
