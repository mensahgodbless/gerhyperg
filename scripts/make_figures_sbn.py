"""
Generate figures from existing result JSONs (seaborn-styled).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



sns.set_theme(
    context="paper",
    style="whitegrid",
    font="DejaVu Sans",
    font_scale=1.0,
    rc={
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.8,
        "grid.color":        "#D0D0D0",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
    },
)


ARCH_PALETTE = sns.color_palette("colorblind", n_colors=10)

# Map each architecture to a stable palette index so 3-branch and 4-branch
ARCH_COLOR_IDX = {
    "main":                0,    # hypergraph blue
    "no_pose":             1,    # orange
    "mean_pool":           2,    # green
    "gat":                 3,    # red
    "holistic":            4,    # purple

    # 4-branch variants share the index of their 3-branch counterpart;
    # plotting code darkens them to distinguish.
    "hypergraph_temporal": 0,
    "no_pose_temporal":    1,
    "mean_pool_temporal":  2,
    "gat_temporal":        3,
    "holistic_temporal":   4,
}

# Modality palette (Set2 — soft, distinct).
MODALITY_PALETTE = sns.color_palette("Set2", n_colors=8)
MODALITY_COLORS = {
    "visual_only":   MODALITY_PALETTE[2],
    "audio_only":    MODALITY_PALETTE[1],
    "scene_only":    MODALITY_PALETTE[7],
    "temporal_only": MODALITY_PALETTE[3],
    "fusion_3":      MODALITY_PALETTE[0],
    "fusion_4":      MODALITY_PALETTE[4],
}


def _darken(rgb_tuple, factor=0.7):
    """Darken an RGB tuple by multiplying each channel by `factor`."""
    return tuple(max(0.0, c * factor) for c in rgb_tuple)


def arch_color(arch: str) -> tuple:
    """Return the seaborn-palette color for an architecture name.
    4-branch variants are darkened versions of their 3-branch counterpart.
    """
    idx = ARCH_COLOR_IDX.get(arch)
    if idx is None:
        return (0.5, 0.5, 0.5)
    base = ARCH_PALETTE[idx]
    return _darken(base, 0.65) if "_temporal" in arch else base


ARCH_LABELS = {
    "main":                "Hypergraph",
    "no_pose":             "No-pose",
    "mean_pool":           "Mean-pool",
    "gat":                 "GAT",
    "holistic":            "Holistic",

    "hypergraph_temporal": "Hypergraph",
    "no_pose_temporal":    "No-pose",
    "mean_pool_temporal":  "Mean-pool",
    "gat_temporal":        "GAT",
    "holistic_temporal":   "Holistic",

    "temporal_only":       "Temporal",
    "visual_only":         "Visual",
    "audio_only":          "Audio",
    "scene_only":          "Scene",
}

CLASS_NAMES = ("positive", "neutral", "negative")

THREE_BRANCH_ORDER = ["main", "no_pose", "mean_pool", "gat", "holistic"]
FOUR_BRANCH_ORDER  = ["hypergraph_temporal", "no_pose_temporal",
                      "mean_pool_temporal", "gat_temporal", "holistic_temporal"]

ARCH_PAIRS = [
    ("main",      "hypergraph_temporal"),
    ("no_pose",   "no_pose_temporal"),
    ("mean_pool", "mean_pool_temporal"),
    ("gat",       "gat_temporal"),
    ("holistic",  "holistic_temporal"),
]
PAIR_LABELS = ["Hypergraph", "No-pose", "Mean-pool", "GAT", "Holistic"]




def load_summary(results_root: Path, dataset: str, ablation: str) -> Optional[dict]:
    sp = results_root / f"{dataset}_{ablation}" / "summary.json"
    if not sp.exists():
        return None
    return json.loads(sp.read_text())


def per_seed_accs(results_root: Path, dataset: str, ablation: str) -> list[float]:
    """Per-seed (VGAF) or per-fold (GECV) accuracies as raw observations."""
    s = load_summary(results_root, dataset, ablation)
    if s is None:
        return []
    if dataset == "vgaf":
        return s.get("per_seed_acc", [])
    else:
        return s.get("per_fold_acc", [])


def build_long_df(
    results_root: Path,
    dataset: str,
    archs: List[str],
) -> pd.DataFrame:
    """Long-format dataframe with one row per (arch, seed/fold) observation."""
    rows = []
    for ab in archs:
        accs = per_seed_accs(results_root, dataset, ab)
        for run_idx, acc in enumerate(accs):
            rows.append({
                "architecture": ab,
                "arch_label": ARCH_LABELS.get(ab, ab),
                "accuracy": acc * 100,
                "run": run_idx,
            })
    return pd.DataFrame(rows)


def load_cross_eval_reports(
    results_root: Path,
    source: str = "vgaf",
    target: str = "gecv",
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    pattern = f"{source}_*/seed_*/cross_eval_{target}/report.json"
    for rp in sorted(results_root.glob(pattern)):
        ablation = rp.parts[-4].replace(f"{source}_", "")
        report = json.loads(rp.read_text())
        out.setdefault(ablation, []).append(report)
    return out


def sum_confusion_matrices(results_root: Path, dataset: str,
                          ablation: str) -> Optional[np.ndarray]:
    run_dir = results_root / f"{dataset}_{ablation}"
    if not run_dir.exists():
        return None
    child_glob = "seed_*/metrics.json" if dataset == "vgaf" else "fold_*/metrics.json"
    subkey = "test_metrics" if dataset == "vgaf" else "val_metrics"
    total: Optional[np.ndarray] = None
    for child in sorted(run_dir.glob(child_glob)):
        m = json.loads(child.read_text())
        cm = m.get(subkey, {}).get("confusion_matrix")
        if cm is None:
            continue
        cm_arr = np.asarray(cm, dtype=int)
        total = cm_arr if total is None else (total + cm_arr)
    return total



def _plot_arch_bars(
    df: pd.DataFrame,
    archs: List[str],
    ylim: tuple,
    baselines: List[tuple],
    ylabel: str,
    errorbar: str,
) -> plt.Figure:
    """Bar chart of accuracy per architecture with error bars."""
    palette = {ARCH_LABELS.get(a, a): arch_color(a) for a in archs}
    order = [ARCH_LABELS.get(a, a) for a in archs]

    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    sns.barplot(
        data=df, x="arch_label", y="accuracy",
        order=order, hue="arch_label", palette=palette,
        errorbar=errorbar, capsize=0.15, err_kws={"linewidth": 1.2, "color": "#222"},
        edgecolor="black", linewidth=0.6, ax=ax,
        legend=False,
    )

    # Mean ± std label above each bar.
    for i, ab in enumerate(archs):
        accs = df[df["architecture"] == ab]["accuracy"]
        if len(accs) == 0:
            continue
        m = accs.mean()
        s = accs.std(ddof=1) if len(accs) > 1 else 0.0
        ax.text(i, m + s + 0.4, f"{m:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#222")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)

    # Baselines as horizontal lines.
    for y, label, color in baselines:
        ax.axhline(y, color=color, linestyle=":", linewidth=1.2, label=label)
    if baselines:
        ax.legend(loc="lower right", framealpha=0.92, fontsize=8,
                  frameon=True, fancybox=False, edgecolor="#aaaaaa")

    return fig


def fig1_main_results(results_root: Path, output_dir: Path, dpi: int,
                      errorbar: str) -> None:
    vgaf_baselines = [
        (70.23, "Wang ICPR'24 (70.23%)", "#888"),
        (81.98, "Kumar ICPR'24 (81.98%)", "#aa3333"),
    ]
    gecv_baselines = [
        (92.90, "Wang ICPR'24 (92.90%)", "#888"),
    ]

    df = build_long_df(results_root, "vgaf", THREE_BRANCH_ORDER)
    fig = _plot_arch_bars(df, THREE_BRANCH_ORDER, (65, 85), vgaf_baselines,
                          "VGAF test accuracy (%)", errorbar)
    _save(fig, output_dir, "fig1a_3branch_vgaf", dpi)

    df = build_long_df(results_root, "gecv", THREE_BRANCH_ORDER)
    fig = _plot_arch_bars(df, THREE_BRANCH_ORDER, (85, 100), gecv_baselines,
                          "GECV cross-validation accuracy (%)", errorbar)
    _save(fig, output_dir, "fig1b_3branch_gecv", dpi)

    df = build_long_df(results_root, "vgaf", FOUR_BRANCH_ORDER)
    fig = _plot_arch_bars(df, FOUR_BRANCH_ORDER, (65, 85), vgaf_baselines,
                          "VGAF test accuracy (%)", errorbar)
    _save(fig, output_dir, "fig1c_4branch_vgaf", dpi)

    df = build_long_df(results_root, "gecv", FOUR_BRANCH_ORDER)
    fig = _plot_arch_bars(df, FOUR_BRANCH_ORDER, (85, 100), gecv_baselines,
                          "GECV cross-validation accuracy (%)", errorbar)
    _save(fig, output_dir, "fig1d_4branch_gecv", dpi)


# Modality contribution — 2 files
def _plot_modality(
    results_root: Path,
    dataset: str,
    best_3branch: str,
    best_4branch: str,
    chance_y: float,
    ylim: tuple,
    ylabel: str,
    errorbar: str,
    legend_loc: str = "upper left",
    legend_bbox: tuple = None,
) -> plt.Figure:
    configs = [
        ("visual_only",  "Visual",        MODALITY_COLORS["visual_only"]),
        ("audio_only",   "Audio",         MODALITY_COLORS["audio_only"]),
        ("scene_only",   "Scene",         MODALITY_COLORS["scene_only"]),
        ("temporal_only","Temporal",      MODALITY_COLORS["temporal_only"]),
        (best_3branch,   "Best 3-branch", MODALITY_COLORS["fusion_3"]),
        (best_4branch,   "Best 4-branch", MODALITY_COLORS["fusion_4"]),
    ]
    rows = []
    palette = {}
    order = []
    for ab, label, color in configs:
        accs = per_seed_accs(results_root, dataset, ab)
        if not accs:
            continue
        for r_i, a in enumerate(accs):
            rows.append({"config": label, "accuracy": a * 100, "run": r_i})
        palette[label] = color
        order.append(label)
    df = pd.DataFrame(rows)
    if df.empty:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    sns.barplot(
        data=df, x="config", y="accuracy",
        order=order, hue="config", palette=palette,
        errorbar=errorbar, capsize=0.15, err_kws={"linewidth": 1.2, "color": "#222"},
        edgecolor="black", linewidth=0.6, ax=ax,
        legend=False,
    )

    for i, label in enumerate(order):
        accs = df[df["config"] == label]["accuracy"]
        m = accs.mean()
        s = accs.std(ddof=1) if len(accs) > 1 else 0.0
        ax.text(i, m + s + 0.6, f"{m:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#222")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=15)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    ax.axhline(chance_y, color="#aa3333", linestyle=":", linewidth=1,
               label=f"Majority-class ({chance_y:.0f}%)")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8,
              frameon=True, fancybox=False, edgecolor="#aaaaaa")

    legend_kwargs = dict(framealpha=0.92, fontsize=8, frameon=True,
                         fancybox=False, edgecolor="#aaaaaa")
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(loc=legend_loc, **legend_kwargs)

    return fig


def fig2_modality(results_root: Path, output_dir: Path, dpi: int,
                  errorbar: str) -> None:
    fig = _plot_modality(
        results_root, "vgaf",
        best_3branch="no_pose",
        best_4branch="gat_temporal",
        chance_y=36.0,
        ylim=(40, 85),
        ylabel="VGAF test accuracy (%)",
        errorbar=errorbar,
        legend_loc="upper left",
    )
    _save(fig, output_dir, "fig2a_modality_vgaf", dpi)

    fig = _plot_modality(
        results_root, "gecv",
        best_3branch="gat",
        best_4branch="holistic_temporal",
        chance_y=49.3,
        ylim=(45, 100),
        ylabel="GECV cross-validation accuracy (%)",
        errorbar=errorbar,
        legend_loc="lower right",
    )
    _save(fig, output_dir, "fig2b_modality_gecv", dpi)


# Face-count stratified (cross-dataset)

def fig3_face_count(results_root: Path, output_dir: Path, dpi: int,
                    errorbar: str) -> None:
    by_ablation = load_cross_eval_reports(results_root, "vgaf", "gecv")
    if not by_ablation:
        print("  No cross-eval reports found — skipping fig 3.")
        return

    archs_to_show = ["main", "gat", "holistic",
                     "hypergraph_temporal", "gat_temporal", "holistic_temporal"]
    arch_pretty = {
        "main":                "Hypergraph (3-br)",
        "gat":                 "GAT (3-br)",
        "holistic":            "Holistic (3-br)",
        "hypergraph_temporal": "Hypergraph (4-br)",
        "gat_temporal":        "GAT (4-br)",
        "holistic_temporal":   "Holistic (4-br)",
    }

    sample_report = next(iter(by_ablation.values()))[0]
    bucket_names = list(sample_report["face_count_stratified"].keys())
    bucket_ns = [sample_report["face_count_stratified"][b]["n"]
                 for b in bucket_names]

    # Build long-form df: one row per (arch, bucket, seed)
    rows = []
    for ab in archs_to_show:
        for r_i, r in enumerate(by_ablation.get(ab, [])):
            for bucket in bucket_names:
                m = r["face_count_stratified"][bucket]
                if m["n"] > 0 and "accuracy" in m:
                    rows.append({
                        "architecture": arch_pretty[ab],
                        "bucket":       bucket,
                        "accuracy":     m["accuracy"] * 100,
                        "run":          r_i,
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    df["bucket"] = pd.Categorical(df["bucket"], categories=bucket_names, ordered=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)

    # Grey-shade small (n<30) buckets in the background.
    for i, n in enumerate(bucket_ns):
        if n < 30:
            ax.axvspan(i - 0.45, i + 0.45, color="#E0E0E0", alpha=0.5, zorder=0)

    palette = {arch_pretty[ab]: arch_color(ab) for ab in archs_to_show}
    dashes = {arch_pretty[ab]: (1, 0) if "_temporal" not in ab else (4, 2)
              for ab in archs_to_show}
    markers = {arch_pretty[ab]: "o" if "_temporal" not in ab else "s"
               for ab in archs_to_show}

    sns.lineplot(
        data=df, x="bucket", y="accuracy", hue="architecture",
        style="architecture",
        palette=palette, dashes=dashes, markers=markers,
        markersize=8, linewidth=2,
        errorbar=errorbar, err_style="bars",
        err_kws={"capsize": 4, "capthick": 1.2, "linewidth": 1.2},
        ax=ax,
    )

    bucket_labels = [f"{b}\n(n={n})" for b, n in zip(bucket_names, bucket_ns)]
    ax.set_xticks(range(len(bucket_names)))
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Mean detected faces per clip")
    ax.set_ylabel("VGAF → GECV transfer accuracy (%)")
    ax.set_ylim(40, 105)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8, ncol=2,
              title="Architecture", title_fontsize=8.5,
              frameon=True, fancybox=False, edgecolor="#aaaaaa")

    ax.text(0.02, 0.97, "Shaded: small bucket (n < 30)",
            transform=ax.transAxes, fontsize=8, color="#555",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.5))

    _save(fig, output_dir, "fig3_face_count", dpi)


# Confusion matrices — 4 files (3 VGAF + 1 GECV)

def _plot_confusion(cm: np.ndarray) -> plt.Figure:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{int(cm[i,j])}\n({cm_norm[i,j]*100:.1f}%)"
                       for j in range(3)] for i in range(3)])

    fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    sns.heatmap(
        cm_norm, annot=annot, fmt="",
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Recall (row-normalized)",
                  "ticks": [0, 0.25, 0.5, 0.75, 1.0],
                  "shrink": 0.85},
        annot_kws={"fontsize": 9, "color": "#222"},
        linewidths=0.4, linecolor="#dddddd",
        square=True, ax=ax,
    )
    # Color text white in dark cells for legibility.
    for i in range(3):
        for j in range(3):
            if cm_norm[i, j] > 0.55:
                ax.texts[i * 3 + j].set_color("white")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=15)
    return fig


def fig4_confusion(results_root: Path, output_dir: Path, dpi: int,
                   errorbar: str) -> None:
    # Three VGAF confusion matrices (main, VGAF winner, VGAF holistic_temporal)
    # plus one GECV confusion matrix (GECV winner = holistic_temporal on GECV).
    # Each tuple is (dataset, ablation, output_filename_stem).
    configs = [
        ("vgaf", "main",              "fig4a_confusion_main_vgaf"),
        ("vgaf", "gat_temporal",      "fig4b_confusion_gat_temporal_vgaf"),
        ("vgaf", "holistic_temporal", "fig4c_confusion_holistic_temporal_vgaf"),
        ("gecv", "holistic_temporal", "fig4d_confusion_holistic_temporal_gecv"),
    ]
    for dataset, ab, fname in configs:
        cm = sum_confusion_matrices(results_root, dataset, ab)
        if cm is None:
            print(f"  no confusion matrix for {dataset}_{ab} — skipping.")
            continue
        fig = _plot_confusion(cm)
        _save(fig, output_dir, fname, dpi)


# Per-architecture temporal lift
def _plot_temporal_lift_single(
    results_root: Path,
    dataset: str,
    ylim: tuple,
    ylabel: str,
    errorbar: str,
) -> plt.Figure:
    """Paired bars (3-br vs 4-br) for each visual aggregation, single dataset."""
    rows = []
    for (arch_3, arch_4), pair_label in zip(ARCH_PAIRS, PAIR_LABELS):
        for a, branches in ((arch_3, "3-branch"),
                            (arch_4, "4-branch (+ temporal)")):
            for acc in per_seed_accs(results_root, dataset, a):
                rows.append({
                    "pair":     pair_label,
                    "branches": branches,
                    "accuracy": acc * 100,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return plt.figure()

    pair_palette = {"3-branch": ARCH_PALETTE[0],
                    "4-branch (+ temporal)": _darken(ARCH_PALETTE[0], 0.55)}

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    sns.barplot(
        data=df, x="pair", y="accuracy",
        order=PAIR_LABELS,
        hue="branches", hue_order=["3-branch", "4-branch (+ temporal)"],
        palette=pair_palette,
        errorbar=errorbar, capsize=0.1,
        err_kws={"linewidth": 1.0, "color": "#222"},
        edgecolor="black", linewidth=0.5,
        ax=ax,
    )

    # Δ annotations above each pair.
    for i, pair_label in enumerate(PAIR_LABELS):
        sub = df[df["pair"] == pair_label]
        if sub.empty:
            continue
        m3 = sub[sub["branches"] == "3-branch"]["accuracy"].mean()
        m4 = sub[sub["branches"] == "4-branch (+ temporal)"]["accuracy"].mean()
        if np.isnan(m3) or np.isnan(m4):
            continue
        s3 = sub[sub["branches"] == "3-branch"]["accuracy"].std(ddof=1)
        s4 = sub[sub["branches"] == "4-branch (+ temporal)"]["accuracy"].std(ddof=1)
        s3 = 0 if np.isnan(s3) else s3
        s4 = 0 if np.isnan(s4) else s4
        delta = m4 - m3
        top = max(m3, m4) + max(s3, s4) + 1.0
        sign = "+" if delta >= 0 else ""
        color = "#1B7A4F" if delta > 0 else "#B23838"
        ax.text(i, top, f"{sign}{delta:.2f}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=15)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8,
              frameon=True, fancybox=False, edgecolor="#aaaaaa",
              title=None)
    return fig


def fig5_temporal_lift(results_root: Path, output_dir: Path, dpi: int,
                       errorbar: str) -> None:
    fig = _plot_temporal_lift_single(
        results_root, "vgaf",
        ylim=(68, 82),
        ylabel="VGAF test accuracy (%)",
        errorbar=errorbar,
    )
    _save(fig, output_dir, "fig5a_temporal_lift_vgaf", dpi)

    fig = _plot_temporal_lift_single(
        results_root, "gecv",
        ylim=(86, 100),
        ylabel="GECV cross-validation accuracy (%)",
        errorbar=errorbar,
    )
    _save(fig, output_dir, "fig5b_temporal_lift_gecv", dpi)




def _save(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}  +  {pdf_path}")



FIGURE_FUNCS = {
    1: ("Main results — 4 files (3-branch × VGAF/GECV, 4-branch × VGAF/GECV)",
        fig1_main_results),
    2: ("Modality decomposition — 2 files (VGAF, GECV)", fig2_modality),
    3: ("Face-count stratified cross-dataset plot — 1 file", fig3_face_count),
    4: ("Confusion matrices — 4 files (3 VGAF + 1 GECV)", fig4_confusion),
    5: ("Per-architecture temporal lift — 2 files (VGAF, GECV)", fig5_temporal_lift),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figures", type=int, nargs="+", default=list(FIGURE_FUNCS),
                        choices=list(FIGURE_FUNCS))
    parser.add_argument("--errorbar", default="ci",
                        choices=["ci", "sd", "se"],
                        help="ci = 95%% bootstrap CI (default), sd = ±1 std, se = ±1 SEM")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise FileNotFoundError(f"No results dir at {args.results_root}")

    # Seaborn errorbar argument: "ci" needs to be a tuple ("ci", 95)
    eb = ("ci", 95) if args.errorbar == "ci" else args.errorbar

    for fi in sorted(args.figures):
        title, func = FIGURE_FUNCS[fi]
        print(f"Figure {fi}: {title}")
        func(args.results_root, args.output_dir, args.dpi, eb)

    print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == "__main__":
    main()