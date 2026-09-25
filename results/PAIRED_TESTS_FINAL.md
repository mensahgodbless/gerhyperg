# Paired Statistical Tests

All tests use **paired/matched samples**: each seed (VGAF) or fold (GECV) produces an accuracy for every architecture, so we compare differences within matched conditions rather than independent groups.

- VGAF: **paired t-test** across 5 seeds (parametric).
- GECV: **Wilcoxon signed-rank** across 10 folds (non-parametric, robust at small n).

## Part A — Omnibus and directional tests

### Friedman omnibus tests

Tests whether the k architectures' accuracies are drawn from the same distribution within a comparison family. A rejection (p < 0.05) justifies the downstream pairwise comparisons; a non-rejection means the per-pair patterns should be treated as exploratory.

| Family | k | n | χ² | p | reject H₀? |
|---|---|---|---|---|---|
| 3-branch (VGAF) | — | 5 | 11.677 | 0.0199* | yes |
| 4-branch (VGAF) | — | 5 | 5.592 | 0.2318 | no |
| Modality (VGAF) | — | 5 | 14.040 | 0.0029** | yes |
| 3-branch (GECV) | — | 10 | 2.711 | 0.6074 | no |
| 4-branch (GECV) | — | 10 | 2.222 | 0.6950 | no |
| Modality (GECV) | — | 10 | 22.598 | 0.0000*** | yes |

### Sign test — does adding temporal help universally?

Across all 10 (architecture × dataset) pairs, count the number where 4-branch mean exceeds 3-branch mean. Under H₀ of no temporal effect, directions should be balanced (binomial(10, 0.5)).

- Positive direction (A > B): **10** of 10
- Negative direction (A < B): 0 of 10
- **Sign test p (two-sided): 0.0020****

| Architecture | mean A | mean B | Δ |
|---|---|---|---|
| Hypergraph (VGAF) | 74.96 | 75.59 | +0.63 |
| No-pose (VGAF) | 75.17 | 76.06 | +0.89 |
| Mean-pool (VGAF) | 74.28 | 75.48 | +1.20 |
| GAT (VGAF) | 74.52 | 76.21 | +1.70 |
| Holistic (VGAF) | 72.19 | 73.29 | +1.10 |
| Hypergraph (GECV) | 91.18 | 93.40 | +2.22 |
| No-pose (GECV) | 90.45 | 92.90 | +2.45 |
| Mean-pool (GECV) | 90.96 | 92.90 | +1.95 |
| GAT (GECV) | 91.42 | 93.15 | +1.73 |
| Holistic (GECV) | 90.95 | 93.66 | +2.71 |

### Sign test — is holistic worst on VGAF across all per-person architectures?

Across 8 paired comparisons (4 per-person architectures × 2 branch counts, all on VGAF), count the number where the per-person mean exceeds holistic. Under H₀ of no architectural difference, directions should be balanced (binomial(8, 0.5)).

- Positive direction (A > B): **8** of 8
- Negative direction (A < B): 0 of 8
- **Sign test p (two-sided): 0.0078****

| Architecture | mean A | mean B | Δ |
|---|---|---|---|
| Hypergraph (3-br) | 74.96 | 72.19 | +2.77 |
| No-pose (3-br) | 75.17 | 72.19 | +2.98 |
| Mean-pool (3-br) | 74.28 | 72.19 | +2.09 |
| GAT (3-br) | 74.52 | 72.19 | +2.32 |
| Hypergraph (4-br) | 75.59 | 73.29 | +2.30 |
| No-pose (4-br) | 76.06 | 73.29 | +2.77 |
| Mean-pool (4-br) | 75.48 | 73.29 | +2.19 |
| GAT (4-br) | 76.21 | 73.29 | +2.92 |

### Bootstrap 95% CI for per-architecture temporal lift

Percentile bootstrap with 10,000 resamples, seed=12345. CI is over the mean per-seed/per-fold lift (4-branch minus 3-branch).

| Dataset | Architecture | mean lift (pp) | 95% CI |
|---|---|---|---|
| VGAF | Hypergraph | +0.63 | [-0.08, +1.28] |
| VGAF | No-pose | +0.89 | [+0.39, +1.33] |
| VGAF | Mean-pool | +1.20 | [+0.47, +1.93] |
| VGAF | GAT | +1.70 | [+0.81, +2.51] |
| VGAF | Holistic | +1.10 | [-0.37, +2.38] |
| GECV | Hypergraph | +2.22 | [+0.24, +4.47] |
| GECV | No-pose | +2.45 | [+0.98, +3.93] |
| GECV | Mean-pool | +1.95 | [+0.96, +3.14] |
| GECV | GAT | +1.73 | [+0.50, +2.96] |
| GECV | Holistic | +2.71 | [+1.48, +3.93] |

## Part B — Pairwise families (per-pair detail)

**Multiple-comparison correction:** Holm-Bonferroni (primary), Bonferroni (supplementary).

**Effect size:** Paired Cohen's d (d_z). |d| ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large.

### VGAF

### 3-branch pairwise — VGAF  (paired t-test, n_pairs=10)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hypergraph | No-pose | 5 | 74.96 | 75.17 | -0.21 | 0.8206 | **1.0000** | 1.0000 | -0.11 | negligible |  |
| Hypergraph | Mean-pool | 5 | 74.96 | 74.28 | +0.68 | 0.4410 | **1.0000** | 1.0000 | +0.38 | small |  |
| Hypergraph | GAT | 5 | 74.96 | 74.52 | +0.44 | 0.5724 | **1.0000** | 1.0000 | +0.27 | small |  |
| Hypergraph | Holistic | 5 | 74.96 | 72.19 | +2.77 | 0.1073 | **0.6437** | 1.0000 | +0.93 | large |  |
| No-pose | Mean-pool | 5 | 75.17 | 74.28 | +0.89 | 0.0185 | **0.1665** | 0.1850 | +1.72 | large |  |
| No-pose | GAT | 5 | 75.17 | 74.52 | +0.65 | 0.1615 | **0.8075** | 1.0000 | +0.77 | medium |  |
| No-pose | Holistic | 5 | 75.17 | 72.19 | +2.98 | 0.0058 | **0.0576** | 0.0576 | +2.41 | large |  |
| Mean-pool | GAT | 5 | 74.28 | 74.52 | -0.23 | 0.6677 | **1.0000** | 1.0000 | -0.21 | small |  |
| Mean-pool | Holistic | 5 | 74.28 | 72.19 | +2.09 | 0.0468 | **0.3308** | 0.4678 | +1.27 | large |  |
| GAT | Holistic | 5 | 74.52 | 72.19 | +2.32 | 0.0413 | **0.3308** | 0.4135 | +1.33 | large |  |

### 4-branch pairwise — VGAF  (paired t-test, n_pairs=10)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hypergraph+T | No-pose+T | 5 | 75.59 | 76.06 | -0.47 | 0.6323 | **1.0000** | 1.0000 | -0.23 | small |  |
| Hypergraph+T | Mean-pool+T | 5 | 75.59 | 75.48 | +0.10 | 0.8528 | **1.0000** | 1.0000 | +0.09 | negligible |  |
| Hypergraph+T | GAT+T | 5 | 75.59 | 76.21 | -0.63 | 0.6027 | **1.0000** | 1.0000 | -0.25 | small |  |
| Hypergraph+T | Holistic+T | 5 | 75.59 | 73.29 | +2.30 | 0.1759 | **1.0000** | 1.0000 | +0.73 | medium |  |
| No-pose+T | Mean-pool+T | 5 | 76.06 | 75.48 | +0.57 | 0.3970 | **1.0000** | 1.0000 | +0.42 | small |  |
| No-pose+T | GAT+T | 5 | 76.06 | 76.21 | -0.16 | 0.8604 | **1.0000** | 1.0000 | -0.08 | negligible |  |
| No-pose+T | Holistic+T | 5 | 76.06 | 73.29 | +2.77 | 0.0442 | **0.4424** | 0.4424 | +1.30 | large |  |
| Mean-pool+T | GAT+T | 5 | 75.48 | 76.21 | -0.73 | 0.4619 | **1.0000** | 1.0000 | -0.36 | small |  |
| Mean-pool+T | Holistic+T | 5 | 75.48 | 73.29 | +2.19 | 0.0958 | **0.8623** | 0.9582 | +0.97 | large |  |
| GAT+T | Holistic+T | 5 | 76.21 | 73.29 | +2.92 | 0.1269 | **1.0000** | 1.0000 | +0.86 | large |  |

### 3-branch vs 4-branch — VGAF  (paired t-test, n_pairs=5)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hypergraph | Hypergraph+T | 5 | 74.96 | 75.59 | -0.63 | 0.1875 | **0.3751** | 0.9377 | -0.71 | medium |  |
| No-pose | No-pose+T | 5 | 75.17 | 76.06 | -0.89 | 0.0310 | **0.1238** | 0.1548 | -1.46 | large |  |
| Mean-pool | Mean-pool+T | 5 | 74.28 | 75.48 | -1.20 | 0.0491 | **0.1473** | 0.2455 | -1.25 | large |  |
| GAT | GAT+T | 5 | 74.52 | 76.21 | -1.70 | 0.0207 | **0.1036** | 0.1036 | -1.66 | large |  |
| Holistic | Holistic+T | 5 | 72.19 | 73.29 | -1.10 | 0.2290 | **0.3751** | 1.0000 | -0.63 | medium |  |

### Modality pairwise — VGAF  (paired t-test, n_pairs=6)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Visual | Audio | 5 | 70.44 | 48.59 | +21.85 | 0.0003 | **0.0013** | 0.0017 | +5.34 | large | ** |
| Visual | Scene | 5 | 70.44 | 68.25 | +2.19 | 0.0638 | **0.1275** | 0.3826 | +1.14 | large |  |
| Visual | Temporal | 5 | 70.44 | 65.48 | +4.96 | 0.0223 | **0.0668** | 0.1336 | +1.62 | large |  |
| Audio | Scene | 5 | 48.59 | 68.25 | -19.66 | 0.0001 | **0.0005** | 0.0005 | -7.36 | large | *** |
| Audio | Temporal | 5 | 48.59 | 65.48 | -16.89 | 0.0003 | **0.0013** | 0.0016 | -5.45 | large | ** |
| Scene | Temporal | 5 | 68.25 | 65.48 | +2.77 | 0.0969 | **0.1275** | 0.5814 | +0.97 | large |  |

### GECV

### 3-branch pairwise — GECV  (wilcoxon, n_pairs=10)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hypergraph | No-pose | 10 | 91.18 | 90.45 | +0.73 | 0.6094 | **1.0000** | 1.0000 | +0.36 | small |  |
| Hypergraph | Mean-pool | 10 | 91.18 | 90.96 | +0.23 | 0.8125 | **1.0000** | 1.0000 | +0.06 | negligible |  |
| Hypergraph | GAT | 10 | 91.18 | 91.42 | -0.24 | 0.8438 | **1.0000** | 1.0000 | -0.10 | negligible |  |
| Hypergraph | Holistic | 10 | 91.18 | 90.95 | +0.23 | 1.0000 | **1.0000** | 1.0000 | +0.09 | negligible |  |
| No-pose | Mean-pool | 10 | 90.45 | 90.96 | -0.51 | 0.6250 | **1.0000** | 1.0000 | -0.18 | negligible |  |
| No-pose | GAT | 10 | 90.45 | 91.42 | -0.97 | 0.5156 | **1.0000** | 1.0000 | -0.47 | small |  |
| No-pose | Holistic | 10 | 90.45 | 90.95 | -0.50 | 0.4375 | **1.0000** | 1.0000 | -0.26 | small |  |
| Mean-pool | GAT | 10 | 90.96 | 91.42 | -0.46 | 0.9531 | **1.0000** | 1.0000 | -0.18 | negligible |  |
| Mean-pool | Holistic | 10 | 90.96 | 90.95 | +0.01 | 1.0000 | **1.0000** | 1.0000 | +0.00 | negligible |  |
| GAT | Holistic | 10 | 91.42 | 90.95 | +0.47 | 0.9531 | **1.0000** | 1.0000 | +0.19 | negligible |  |

### 4-branch pairwise — GECV  (wilcoxon, n_pairs=10)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hypergraph+T | No-pose+T | 10 | 93.40 | 92.90 | +0.51 | 0.5000 | **1.0000** | 1.0000 | +0.22 | small |  |
| Hypergraph+T | Mean-pool+T | 10 | 93.40 | 92.90 | +0.50 | 0.2500 | **1.0000** | 1.0000 | +0.32 | small |  |
| Hypergraph+T | GAT+T | 10 | 93.40 | 93.15 | +0.26 | 0.5000 | **1.0000** | 1.0000 | +0.14 | negligible |  |
| Hypergraph+T | Holistic+T | 10 | 93.40 | 93.66 | -0.26 | 0.6250 | **1.0000** | 1.0000 | -0.09 | negligible |  |
| No-pose+T | Mean-pool+T | 10 | 92.90 | 92.90 | -0.01 | 1.0000 | **1.0000** | 1.0000 | -0.00 | negligible |  |
| No-pose+T | GAT+T | 10 | 92.90 | 93.15 | -0.25 | 1.0000 | **1.0000** | 1.0000 | -0.10 | negligible |  |
| No-pose+T | Holistic+T | 10 | 92.90 | 93.66 | -0.76 | 0.3047 | **1.0000** | 1.0000 | -0.23 | small |  |
| Mean-pool+T | GAT+T | 10 | 92.90 | 93.15 | -0.24 | 1.0000 | **1.0000** | 1.0000 | -0.13 | negligible |  |
| Mean-pool+T | Holistic+T | 10 | 92.90 | 93.66 | -0.76 | 0.3438 | **1.0000** | 1.0000 | -0.27 | small |  |
| GAT+T | Holistic+T | 10 | 93.15 | 93.66 | -0.51 | 0.4609 | **1.0000** | 1.0000 | -0.17 | negligible |  |

### 3-branch vs 4-branch — GECV  (wilcoxon, n_pairs=5)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hypergraph | Hypergraph+T | 10 | 91.18 | 93.40 | -2.22 | 0.1250 | **0.1250** | 0.6250 | -0.59 | medium |  |
| No-pose | No-pose+T | 10 | 90.45 | 92.90 | -2.45 | 0.0312 | **0.1250** | 0.1562 | -0.95 | large |  |
| Mean-pool | Mean-pool+T | 10 | 90.96 | 92.90 | -1.95 | 0.0312 | **0.1250** | 0.1562 | -1.02 | large |  |
| GAT | GAT+T | 10 | 91.42 | 93.15 | -1.73 | 0.0625 | **0.1250** | 0.3125 | -0.85 | large |  |
| Holistic | Holistic+T | 10 | 90.95 | 93.66 | -2.71 | 0.0156 | **0.0781** | 0.0781 | -1.26 | large |  |

### Modality pairwise — GECV  (wilcoxon, n_pairs=6)

Significance markers reflect **Holm-Bonferroni** corrected p: `*` p<0.05, `**` p<0.01, `***` p<0.001.

| A | B | n | mean A | mean B | Δ (A−B) | p (raw) | **p (Holm)** | p (Bonf) | Cohen's d | effect | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Visual | Audio | 10 | 90.69 | 61.07 | +29.63 | 0.0020 | **0.0117** | 0.0117 | +3.23 | large | * |
| Visual | Scene | 10 | 90.69 | 93.37 | -2.68 | 0.0469 | **0.1406** | 0.2812 | -0.86 | large |  |
| Visual | Temporal | 10 | 90.69 | 91.41 | -0.72 | 0.5859 | **0.5859** | 1.0000 | -0.17 | negligible |  |
| Audio | Scene | 10 | 61.07 | 93.37 | -32.30 | 0.0020 | **0.0117** | 0.0117 | -3.66 | large | * |
| Audio | Temporal | 10 | 61.07 | 91.41 | -30.35 | 0.0020 | **0.0117** | 0.0117 | -3.83 | large | * |
| Scene | Temporal | 10 | 93.37 | 91.41 | +1.96 | 0.0547 | **0.1406** | 0.3281 | +0.87 | large |  |

### Pairwise summary

Counts of significant pairs (after correction) and large-effect-size pairs (|d| ≥ 0.8) per family.

| Family | Pairs | Holm α<0.05 | Holm α<0.01 | Bonf α<0.05 | \|d\| ≥ 0.8 |
|---|---|---|---|---|---|
| 3-branch pairwise | 20 | 0 | 0 | 0 | 5 |
| 4-branch pairwise | 20 | 0 | 0 | 0 | 3 |
| 3-branch vs 4-branch | 10 | 0 | 0 | 0 | 7 |
| Modality pairwise | 12 | 6 | 3 | 6 | 11 |

## Reading these tables

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
