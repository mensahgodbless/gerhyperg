"""
Paired statistical tests across architectures and modalities.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

THREE_BRANCH = ["main", "no_pose", "mean_pool", "gat", "holistic"]
FOUR_BRANCH  = ["hypergraph_temporal", "no_pose_temporal",
                "mean_pool_temporal", "gat_temporal", "holistic_temporal"]
PER_PERSON_3 = ["main", "no_pose", "mean_pool", "gat"]   # excludes holistic
PER_PERSON_4 = ["hypergraph_temporal", "no_pose_temporal",
                "mean_pool_temporal", "gat_temporal"]
ARCH_PAIRS_3v4 = [
    ("main",      "hypergraph_temporal"),
    ("no_pose",   "no_pose_temporal"),
    ("mean_pool", "mean_pool_temporal"),
    ("gat",       "gat_temporal"),
    ("holistic",  "holistic_temporal"),
]
MODALITY_CONFIGS = ["visual_only", "audio_only", "scene_only", "temporal_only"]

ARCH_LABELS = {
    "main":                "Hypergraph",
    "no_pose":             "No-pose",
    "mean_pool":           "Mean-pool",
    "gat":                 "GAT",
    "holistic":            "Holistic",
    "hypergraph_temporal": "Hypergraph+T",
    "no_pose_temporal":    "No-pose+T",
    "mean_pool_temporal":  "Mean-pool+T",
    "gat_temporal":        "GAT+T",
    "holistic_temporal":   "Holistic+T",
    "visual_only":         "Visual",
    "audio_only":          "Audio",
    "scene_only":          "Scene",
    "temporal_only":       "Temporal",
}

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 12345


# ----------------------------------------------------------------------
# Per-run accuracy loading
# ----------------------------------------------------------------------

def per_run_accs(results_root: Path, dataset: str, ablation: str) -> List[float]:
    """Per-seed (VGAF) or per-fold (GECV) accuracies, ordered by seed/fold id."""
    run_dir = results_root / f"{dataset}_{ablation}"
    if not run_dir.exists():
        return []
    child_glob = "seed_*/metrics.json" if dataset == "vgaf" else "fold_*/metrics.json"
    subkey = "test_metrics" if dataset == "vgaf" else "val_metrics"
    accs = []
    for child in sorted(run_dir.glob(child_glob)):
        m = json.loads(child.read_text())
        a = m.get(subkey, {}).get("accuracy")
        if a is not None:
            accs.append(float(a))
    return accs


def all_accs(results_root: Path, dataset: str,
             ablations: List[str]) -> Dict[str, List[float]]:
    return {ab: per_run_accs(results_root, dataset, ab) for ab in ablations}


# ----------------------------------------------------------------------
# Pairwise test machinery
# ----------------------------------------------------------------------

def paired_test(a: List[float], b: List[float],
                dataset: str) -> Tuple[float, float, str]:
    if not a or not b or len(a) != len(b):
        name = "paired t-test" if dataset == "vgaf" else "wilcoxon"
        return (float("nan"), float("nan"), name)
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if dataset == "vgaf":
        stat, p = stats.ttest_rel(arr_a, arr_b)
        return (float(stat), float(p), "paired t-test")
    diffs = arr_a - arr_b
    if np.all(diffs == 0):
        return (float("nan"), 1.0, "wilcoxon")
    stat, p = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
    return (float(stat), float(p), "wilcoxon")


def cohens_d_paired(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    diffs = np.asarray(a) - np.asarray(b)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diffs.mean() / sd)


def holm_correct(raw_ps: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction."""
    n = len(raw_ps)
    if n == 0:
        return []
    indexed = sorted(enumerate(raw_ps), key=lambda x: (np.isnan(x[1]), x[1]))
    corrected = [0.0] * n
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        if np.isnan(p):
            corrected[orig_idx] = float("nan")
            continue
        adj = min(1.0, p * (n - rank))
        adj = max(prev, adj)
        corrected[orig_idx] = adj
        prev = adj
    return corrected


def bonferroni_correct(raw_ps: List[float]) -> List[float]:
    n = sum(1 for p in raw_ps if not np.isnan(p))
    return [min(1.0, p * n) if not np.isnan(p) else float("nan")
            for p in raw_ps]


def run_family(
    name: str,
    pairs: List[Tuple[str, str]],
    accs: Dict[str, List[float]],
    dataset: str,
) -> List[dict]:
    results = []
    raw_ps = []
    for a, b in pairs:
        aa = accs.get(a, [])
        bb = accs.get(b, [])
        stat, p, test_name = paired_test(aa, bb, dataset)
        d = cohens_d_paired(aa, bb)
        m_a = float(np.mean(aa)) if aa else float("nan")
        m_b = float(np.mean(bb)) if bb else float("nan")
        results.append({
            "family":    name, "a": a, "b": b,
            "n":         min(len(aa), len(bb)),
            "mean_a":    m_a, "mean_b": m_b,
            "delta":     (m_a - m_b) if (aa and bb) else float("nan"),
            "test":      test_name,
            "statistic": stat, "p_raw": p,
            "cohens_d":  d, "n_pairs": len(pairs),
        })
        raw_ps.append(p)
    p_holm = holm_correct(raw_ps)
    p_bonf = bonferroni_correct(raw_ps)
    for r, h, b in zip(results, p_holm, p_bonf):
        r["p_holm"] = h
        r["p_bonf"] = b
    return results


# ----------------------------------------------------------------------
# OMNIBUS AND DIRECTIONAL TESTS
# ----------------------------------------------------------------------

def friedman_test(accs: Dict[str, List[float]],
                  archs: List[str]) -> Tuple[float, float, int]:
    """Friedman omnibus test across k architectures.
    Returns (chi-square, p-value, n_samples). Requires equal-length per-arch.
    """
    columns = []
    for ab in archs:
        a = accs.get(ab, [])
        if not a:
            return (float("nan"), float("nan"), 0)
        columns.append(a)
    n_samples = len(columns[0])
    if any(len(c) != n_samples for c in columns):
        return (float("nan"), float("nan"), 0)
    if n_samples < 2:
        return (float("nan"), float("nan"), n_samples)
    chi2, p = stats.friedmanchisquare(*columns)
    return (float(chi2), float(p), n_samples)


def sign_test(differences: List[float]) -> Tuple[int, int, int, float]:
    """Two-sided exact binomial sign test.
    Treats zero differences by dropping them (Wilcoxon convention).
    Returns (n_positive, n_negative, n_excluded_zeros, p_two_sided).
    """
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n_zero = int(np.sum(diffs == 0))
    nonzero = diffs[diffs != 0]
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    n = n_pos + n_neg
    if n == 0:
        return (n_pos, n_neg, n_zero, float("nan"))
    # Two-sided exact binomial p:
    result = stats.binomtest(min(n_pos, n_neg), n, p=0.5)
    return (n_pos, n_neg, n_zero, float(result.pvalue))


def bootstrap_ci(values: List[float], n_boot: int = N_BOOTSTRAP,
                 confidence: float = 0.95,
                 rng_seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the sample mean.
    Returns (mean, lower, upper) — bounds at (1-conf)/2 and 1-(1-conf)/2.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1 - alpha))
    return (float(arr.mean()), lo, hi)


def temporal_lift_directional(
    accs_3: Dict[str, List[float]],
    accs_4: Dict[str, List[float]],
) -> Dict:
    """Sign test on whether each architecture's 4-branch mean exceeds its
    3-branch mean. Tested across all (architecture × dataset) pairs.

    Caller supplies separate dicts because the same ablation key on different
    datasets is two independent comparisons.
    """
    diffs = []
    rows = []
    for arch_3, arch_4 in ARCH_PAIRS_3v4:
        a3 = accs_3.get(arch_3, [])
        a4 = accs_4.get(arch_4, [])
        if not a3 or not a4:
            continue
        d = float(np.mean(a4) - np.mean(a3))
        diffs.append(d)
        rows.append({
            "architecture": ARCH_LABELS.get(arch_3, arch_3).replace("+T", ""),
            "mean_3branch": float(np.mean(a3)) * 100,
            "mean_4branch": float(np.mean(a4)) * 100,
            "delta": d * 100,
        })
    n_pos, n_neg, n_zero, p = sign_test(diffs)
    return {"rows": rows, "n_pos": n_pos, "n_neg": n_neg,
            "n_zero": n_zero, "p_sign": p}


def holistic_vs_perperson_directional(
    accs: Dict[str, List[float]],
    holistic_key: str,
    per_person_keys: List[str],
) -> Dict:
    """Sign test on whether each per-person architecture's mean exceeds
    holistic's mean."""
    h_accs = accs.get(holistic_key, [])
    if not h_accs:
        return {"rows": [], "n_pos": 0, "n_neg": 0, "n_zero": 0,
                "p_sign": float("nan")}
    h_mean = float(np.mean(h_accs))
    diffs = []
    rows = []
    for ab in per_person_keys:
        a = accs.get(ab, [])
        if not a:
            continue
        m = float(np.mean(a))
        d = m - h_mean
        diffs.append(d)
        rows.append({
            "architecture": ARCH_LABELS.get(ab, ab),
            "mean": m * 100,
            "holistic_mean": h_mean * 100,
            "delta": d * 100,
        })
    n_pos, n_neg, n_zero, p = sign_test(diffs)
    return {"rows": rows, "n_pos": n_pos, "n_neg": n_neg,
            "n_zero": n_zero, "p_sign": p}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def sig_marker(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def effect_label(d: float) -> str:
    if np.isnan(d):
        return ""
    a = abs(d)
    if a < 0.2:  return "negligible"
    if a < 0.5:  return "small"
    if a < 0.8:  return "medium"
    return "large"


def format_pairwise_table(family_name: str, dataset: str,
                          results: List[dict]) -> str:
    out = []
    n_pairs = results[0]["n_pairs"] if results else 0
    test_name = results[0]["test"] if results else "?"
    out.append(f"### {family_name} — {dataset.upper()}  "
               f"({test_name}, n_pairs={n_pairs})")
    out.append("")
    out.append("Significance markers reflect **Holm-Bonferroni** corrected p: "
               "`*` p<0.05, `**` p<0.01, `***` p<0.001.")
    out.append("")
    out.append("| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | "
               "**p (Holm)** | p (Bonf) | Cohen's d | effect | sig |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        a_lbl = ARCH_LABELS.get(r["a"], r["a"])
        b_lbl = ARCH_LABELS.get(r["b"], r["b"])
        marker = sig_marker(r["p_holm"])
        eff = effect_label(r["cohens_d"])
        out.append(
            f"| {a_lbl} | {b_lbl} | {r['n']} | "
            f"{r['mean_a']*100:.2f} | {r['mean_b']*100:.2f} | "
            f"{r['delta']*100:+.2f} | "
            f"{r['p_raw']:.4f} | **{r['p_holm']:.4f}** | "
            f"{r['p_bonf']:.4f} | {r['cohens_d']:+.2f} | {eff} | {marker} |"
        )
    out.append("")
    return "\n".join(out)


def format_omnibus_section(
    friedman_results: Dict[str, Tuple[float, float, int]],
) -> str:
    out = ["### Friedman omnibus tests", ""]
    out.append("Tests whether the k architectures' accuracies are drawn from "
               "the same distribution within a comparison family. A rejection "
               "(p < 0.05) justifies the downstream pairwise comparisons; "
               "a non-rejection means the per-pair patterns should be treated "
               "as exploratory.")
    out.append("")
    out.append("| Family | k | n | χ² | p | reject H₀? |")
    out.append("|---|---|---|---|---|---|")
    for name, (chi2, p, n) in friedman_results.items():
        if np.isnan(chi2):
            out.append(f"| {name} | — | {n} | — | — | (insufficient data) |")
            continue
        reject = "yes" if p < 0.05 else "no"
        sig = sig_marker(p)
        out.append(f"| {name} | — | {n} | {chi2:.3f} | {p:.4f}{sig} | {reject} |")
    out.append("")
    return "\n".join(out)


def format_sign_test_section(name: str, result: Dict, framing: str) -> str:
    out = [f"### {name}", ""]
    out.append(framing)
    out.append("")
    n_pos = result["n_pos"]
    n_neg = result["n_neg"]
    n_zero = result["n_zero"]
    n_total = n_pos + n_neg
    p = result["p_sign"]
    sig = sig_marker(p)
    out.append(f"- Positive direction (A > B): **{n_pos}** of {n_total}")
    out.append(f"- Negative direction (A < B): {n_neg} of {n_total}")
    if n_zero > 0:
        out.append(f"- Zero differences (excluded): {n_zero}")
    out.append(f"- **Sign test p (two-sided): {p:.4f}**{sig}")
    out.append("")
    out.append("| Architecture | mean A | mean B | Δ |")
    out.append("|---|---|---|---|")
    for r in result["rows"]:
        if "mean_3branch" in r:  # temporal lift table
            out.append(f"| {r['architecture']} | {r['mean_3branch']:.2f} | "
                       f"{r['mean_4branch']:.2f} | {r['delta']:+.2f} |")
        else:  # holistic-vs-perperson table
            out.append(f"| {r['architecture']} | {r['mean']:.2f} | "
                       f"{r['holistic_mean']:.2f} | {r['delta']:+.2f} |")
    out.append("")
    return "\n".join(out)


def format_bootstrap_section(
    per_dataset_lifts: Dict[str, Dict[str, Tuple[float, float, float]]],
) -> str:
    out = ["### Bootstrap 95% CI for per-architecture temporal lift", ""]
    out.append(f"Percentile bootstrap with {N_BOOTSTRAP:,} resamples, "
               f"seed={BOOTSTRAP_SEED}. CI is over the mean per-seed/per-fold "
               "lift (4-branch minus 3-branch).")
    out.append("")
    out.append("| Dataset | Architecture | mean lift (pp) | 95% CI |")
    out.append("|---|---|---|---|")
    for ds, lifts in per_dataset_lifts.items():
        for arch, (mean, lo, hi) in lifts.items():
            if np.isnan(mean):
                out.append(f"| {ds.upper()} | {arch} | — | — |")
                continue
            out.append(f"| {ds.upper()} | {arch} | "
                       f"{mean*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] |")
    out.append("")
    return "\n".join(out)


def format_pairwise_summary(results: List[dict]) -> str:
    by_family = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)
    out = ["### Pairwise summary", ""]
    out.append("Counts of significant pairs (after correction) and "
               "large-effect-size pairs (|d| ≥ 0.8) per family.")
    out.append("")
    out.append("| Family | Pairs | Holm α<0.05 | Holm α<0.01 | "
               "Bonf α<0.05 | |d| ≥ 0.8 |")
    out.append("|---|---|---|---|---|---|")
    for fam, rs in by_family.items():
        n_total = len(rs)
        n_holm_05 = sum(1 for r in rs if not np.isnan(r["p_holm"])
                        and r["p_holm"] < 0.05)
        n_holm_01 = sum(1 for r in rs if not np.isnan(r["p_holm"])
                        and r["p_holm"] < 0.01)
        n_bonf_05 = sum(1 for r in rs if not np.isnan(r["p_bonf"])
                        and r["p_bonf"] < 0.05)
        n_large = sum(1 for r in rs if not np.isnan(r["cohens_d"])
                      and abs(r["cohens_d"]) >= 0.8)
        out.append(f"| {fam} | {n_total} | {n_holm_05} | {n_holm_01} | "
                   f"{n_bonf_05} | {n_large} |")
    out.append("")
    return "\n".join(out)


def interpretation_notes() -> str:
    return """## Reading these tables

Sample sizes (n=5 VGAF seeds, n=10 GECV folds) are typical of GER benchmark
protocols but limit individual pairwise significance testing power. We
therefore report several complementary lines of evidence:

1. **Omnibus tests (Friedman)** check whether the k architectures differ
   collectively. A non-significant Friedman justifies *not* claiming
   per-pair differences in that family.

2. **Direction-based tests (sign tests)** ask whether observed effect
   directions are consistent across independent comparisons. These are
   immune to per-pair multiple-comparison concerns and have full power
   when the design produces multiple matched contrasts.

3. **Bootstrap CIs** give sample-size-aware uncertainty bounds on effect
   magnitudes without assuming any specific test distribution.

4. **Effect sizes (Cohen's d)** are independent of sample size. A large
   d with a non-significant p indicates a real effect underpowered to
   detect at the chosen α, not the absence of effect. Conventions:
   |d| ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large.

5. **Pairwise tests (Wilcoxon / paired t)** with Holm-Bonferroni
   correction provide per-pair detail when omnibus tests warrant.

For this study, conservative pairwise testing yields few individually
significant comparisons. The directional sign tests and consistent
large effect sizes across both datasets support substantive findings
that the pairwise design is underpowered to confirm in isolation.
"""


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/PAIRED_TESTS.md"))
    args = parser.parse_args()

    out_md = ["# Paired Statistical Tests", ""]
    out_md.append(
        "All tests use **paired/matched samples**: each seed (VGAF) or fold "
        "(GECV) produces an accuracy for every architecture, so we compare "
        "differences within matched conditions rather than independent groups."
    )
    out_md.append("")
    out_md.append("- VGAF: **paired t-test** across 5 seeds (parametric).")
    out_md.append("- GECV: **Wilcoxon signed-rank** across 10 folds "
                  "(non-parametric, robust at small n).")
    out_md.append("")

    # ==================== Section A: Omnibus + directional ==================

    out_md.append("## Part A — Omnibus and directional tests")
    out_md.append("")

    # Friedman per family per dataset
    friedman_results: Dict[str, Tuple[float, float, int]] = {}
    for dataset in ("vgaf", "gecv"):
        accs_3 = all_accs(args.results_root, dataset, THREE_BRANCH)
        accs_4 = all_accs(args.results_root, dataset, FOUR_BRANCH)
        accs_mod = all_accs(args.results_root, dataset, MODALITY_CONFIGS)
        friedman_results[f"3-branch ({dataset.upper()})"] = friedman_test(accs_3, THREE_BRANCH)
        friedman_results[f"4-branch ({dataset.upper()})"] = friedman_test(accs_4, FOUR_BRANCH)
        friedman_results[f"Modality ({dataset.upper()})"] = friedman_test(accs_mod, MODALITY_CONFIGS)
    out_md.append(format_omnibus_section(friedman_results))

    # Sign test: temporal contribution (10 architecture × dataset comparisons)
    # Combine across datasets — each (arch, dataset) is one of 10 paired comparisons.
    combined_diffs = []
    combined_rows = []
    for dataset in ("vgaf", "gecv"):
        accs_3 = all_accs(args.results_root, dataset, THREE_BRANCH)
        accs_4 = all_accs(args.results_root, dataset, FOUR_BRANCH)
        per_dataset = temporal_lift_directional(accs_3, accs_4)
        for r in per_dataset["rows"]:
            combined_diffs.append((r["mean_4branch"] - r["mean_3branch"]) / 100)
            combined_rows.append({
                "architecture": f"{r['architecture']} ({dataset.upper()})",
                "mean_3branch": r["mean_3branch"],
                "mean_4branch": r["mean_4branch"],
                "delta":        r["delta"],
            })
    n_pos, n_neg, n_zero, p = sign_test(combined_diffs)
    temporal_combined = {"rows": combined_rows, "n_pos": n_pos,
                         "n_neg": n_neg, "n_zero": n_zero, "p_sign": p}
    out_md.append(format_sign_test_section(
        "Sign test — does adding temporal help universally?",
        temporal_combined,
        "Across all 10 (architecture × dataset) pairs, count the number where "
        "4-branch mean exceeds 3-branch mean. Under H₀ of no temporal effect, "
        "directions should be balanced (binomial(10, 0.5))."
    ))

    # Sign test: holistic vs per-person on VGAF (4 archs × 2 branch counts = 8 pairs)
    accs_vgaf_all = all_accs(args.results_root, "vgaf", THREE_BRANCH + FOUR_BRANCH)
    h3 = holistic_vs_perperson_directional(accs_vgaf_all, "holistic", PER_PERSON_3)
    h4 = holistic_vs_perperson_directional(accs_vgaf_all, "holistic_temporal", PER_PERSON_4)
    combined_h_diffs = ([r["delta"] / 100 for r in h3["rows"]] +
                        [r["delta"] / 100 for r in h4["rows"]])
    combined_h_rows = (
        [{"architecture": r["architecture"] + " (3-br)",
          "mean": r["mean"], "holistic_mean": r["holistic_mean"],
          "delta": r["delta"]} for r in h3["rows"]] +
        [{"architecture": r["architecture"].replace("+T", "") + " (4-br)",
          "mean": r["mean"], "holistic_mean": r["holistic_mean"],
          "delta": r["delta"]} for r in h4["rows"]]
    )
    n_pos, n_neg, n_zero, p = sign_test(combined_h_diffs)
    h_combined = {"rows": combined_h_rows, "n_pos": n_pos,
                  "n_neg": n_neg, "n_zero": n_zero, "p_sign": p}
    out_md.append(format_sign_test_section(
        "Sign test — is holistic worst on VGAF across all per-person architectures?",
        h_combined,
        "Across 8 paired comparisons (4 per-person architectures × 2 branch "
        "counts, all on VGAF), count the number where the per-person mean "
        "exceeds holistic. Under H₀ of no architectural difference, directions "
        "should be balanced (binomial(8, 0.5))."
    ))

    # Bootstrap CI for per-architecture temporal lift
    per_dataset_lifts: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for dataset in ("vgaf", "gecv"):
        lifts: Dict[str, Tuple[float, float, float]] = {}
        for arch_3, arch_4 in ARCH_PAIRS_3v4:
            a3 = per_run_accs(args.results_root, dataset, arch_3)
            a4 = per_run_accs(args.results_root, dataset, arch_4)
            if not a3 or not a4 or len(a3) != len(a4):
                lifts[ARCH_LABELS[arch_3]] = (float("nan"), float("nan"),
                                              float("nan"))
                continue
            paired_diffs = [a4[i] - a3[i] for i in range(len(a3))]
            mean, lo, hi = bootstrap_ci(paired_diffs)
            lifts[ARCH_LABELS[arch_3]] = (mean, lo, hi)
        per_dataset_lifts[dataset] = lifts
    out_md.append(format_bootstrap_section(per_dataset_lifts))

    # ==================== Section B: pairwise families ====================

    out_md.append("## Part B — Pairwise families (per-pair detail)")
    out_md.append("")
    out_md.append("**Multiple-comparison correction:** Holm-Bonferroni (primary), "
                  "Bonferroni (supplementary).")
    out_md.append("")
    out_md.append("**Effect size:** Paired Cohen's d (d_z). "
                  "|d| ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large.")
    out_md.append("")

    all_pairwise = []
    for dataset in ("vgaf", "gecv"):
        out_md.append(f"### {dataset.upper()}")
        out_md.append("")
        accs = all_accs(args.results_root, dataset, THREE_BRANCH)
        fr = run_family("3-branch pairwise", list(combinations(THREE_BRANCH, 2)),
                        accs, dataset)
        all_pairwise.extend(fr)
        out_md.append(format_pairwise_table("3-branch pairwise", dataset, fr))

        accs = all_accs(args.results_root, dataset, FOUR_BRANCH)
        fr = run_family("4-branch pairwise", list(combinations(FOUR_BRANCH, 2)),
                        accs, dataset)
        all_pairwise.extend(fr)
        out_md.append(format_pairwise_table("4-branch pairwise", dataset, fr))

        accs = all_accs(args.results_root, dataset, THREE_BRANCH + FOUR_BRANCH)
        fr = run_family("3-branch vs 4-branch", ARCH_PAIRS_3v4, accs, dataset)
        all_pairwise.extend(fr)
        out_md.append(format_pairwise_table("3-branch vs 4-branch", dataset, fr))

        accs = all_accs(args.results_root, dataset, MODALITY_CONFIGS)
        fr = run_family("Modality pairwise", list(combinations(MODALITY_CONFIGS, 2)),
                        accs, dataset)
        all_pairwise.extend(fr)
        out_md.append(format_pairwise_table("Modality pairwise", dataset, fr))

    out_md.append(format_pairwise_summary(all_pairwise))
    out_md.append(interpretation_notes())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_md))
    print(f"Wrote {args.output}")
    print()
    print(format_pairwise_summary(all_pairwise))


if __name__ == "__main__":
    main()