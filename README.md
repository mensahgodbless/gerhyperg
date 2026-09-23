# When Does Per-Person Modeling Matter? A Controlled Study of Multimodal Group Affect Recognition Across Acquisition Regimes.

Code and reference results for the NCMMSC2026 paper (Springer CCIS, to appear).

The study holds the feature backbones, fusion, training procedure and evaluation
protocol fixed and varies **only** the per-person visual aggregation strategy,
across five variants, two datasets (VGAF, GECV), three- and four-branch
configurations, multi-seed and cross-validated evaluation, and bidirectional
zero-shot transfer.

## Findings

- Across five aggregation strategies, **no per-person mechanism measurably beats
  a masked per-frame mean.** Soft hypergraph, graph attention and mean pooling
  are mutually indistinguishable.
- The advantage of structured aggregators over global holistic pooling on
  in-the-wild VGAF comes mainly from **learned temporal pooling over frames**,
  not from modelling persons. A parameter-matched per-frame mean with uniform
  temporal pooling is indistinguishable from holistic pooling.
- Complexity is not free: hypergraph and graph attention carry 27–36% more
  visual-branch parameters and 1.5–1.6x the FLOPs per clip of mean pooling, for
  no accuracy gain.
- On controlled-acquisition GECV, no aggregator is distinguishable from any
  other.
- Cross-dataset transfer **reverses its ordering with direction**. The larger
  absolute loss in the forward direction reflects target difficulty rather than
  training-set breadth.

Best results: 76.21% on VGAF (GAT, 4-branch) and 93.66% on GECV (Holistic, 4-branch).

## Repository layout

```
configs/            base + per-dataset configs; ablations/ selects variants
data/               datasets, preprocessing, clip discovery
models/             feature projector, aggregation encoders, classifier
training/           losses, metrics, splits, train/eval utilities
scripts/            extraction, training, evaluation, aggregation, statistics
results/            reference outputs: summary.json, resolved configs, metrics
evaluations/        transfer evaluation JSONs behind Table 3
stats/              derived face-count statistics used for stratified analysis
```

Checkpoints are **not** in this repository (about 2.3 GB). DOI will be provided later.

## Installation

Python 3.10. Install PyTorch for your CUDA version first (the paper used torch 2.9.0 with CUDA 12.6):

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

That is enough to **train and evaluate on cached features**.
To rebuild the feature cache from raw video:

```bash
pip install -r requirements-extract.txt
pip install --no-deps facenet-pytorch==2.5.3   # pins an older torch
```

Feature extraction also needs `ffmpeg` on PATH.

All experiments in the paper ran on a single NVIDIA RTX 3080 (10 GB).

## Datasets

Neither dataset is redistributable; request access from the original authors.

- **VGAF** (Sharma et al., IEEE TAFFC 2023). We use Protocol A, the official
  train (2,661 clips) and validation (766 clips) splits. The test set is not
  public.
- **GECV** (Quach et al., Pattern Recognition 2022). 408 of the 627 clips are
  publicly available. Evaluated with stratified 10-fold cross-validation
  (`fold_seed: 42`).

Set the paths in `configs/vgaf.yaml` and `configs/gecv.yaml`, which ship with
`/path/to/datasets/...` placeholders, or override per run:

```bash
python scripts/train_vgaf.py --override data.cache_root=/your/cache/vgaf
```

## Pretrained backbones

All backbones are frozen. None are redistributed here.

| Role | Model | Source |
|---|---|---|
| Face embedding (1408-d) | HSEmotion `enet_b2_8` | HSE-asavchenko/face-emotion-recognition |
| Face detection | MTCNN | `facenet-pytorch` (auto-download) |
| Body / scene (512-d) | CLIP ViT-B/16 | `open_clip_torch` (auto-download) |
| Pose (64-d) | YOLOv8x-pose | `ultralytics` (auto-download) |
| Audio (1024-d) | HuBERT-Large | Hugging Face (see `--hubert_model` default) |
| Temporal (768-d) | `facebook/timesformer-base-finetuned-k400` | Hugging Face |

## Feature extraction

Run once per dataset, in this order. Each script writes per-clip HDF5 into
`<cache_root>/<modality>/<clip_id>.h5` and skips existing files, so it is
resumable.

```bash
# 1. Visual: 15 uniformly sampled frames, 2048-d per person
#    (face 1408 | body 512 | bbox 64 | pose 64)
python scripts/extract_visual_features.py --dataset vgaf \
    --video_root /path/to/datasets/VGAF/Train \
    --output_dir /path/to/cache/vgaf/visual \
    --config configs/vgaf.yaml --use_tracker --gpu 0

# 2. Audio: HuBERT-Large, mean-pooled over time
python scripts/extract_audio_features.py --dataset vgaf \
    --video_root /path/to/datasets/VGAF \
    --output_dir /path/to/cache/vgaf/audio --device cuda

# 3. Scene: CLIP ViT-B/16 on whole frames, mean-pooled over 15 frames
python scripts/extract_scene_features.py --dataset vgaf \
    --video_root /path/to/datasets/VGAF \
    --output_dir /path/to/cache/vgaf/scene --num_frames 15 --device cuda

# 4. Temporal: TimeSformer [CLS], 8 frames (the K400 pretraining value)
python scripts/extract_temporal.py --dataset vgaf \
    --video_root /path/to/datasets/VGAF \
    --cache_root /path/to/cache/vgaf --gpu 0
```

`--use_tracker` is required for step 1: pose keypoints are IoU-matched to the
tracker's canonical persons. Repeat all four with `--dataset gecv`.

Each script accepts `--shard K/N` to split work across GPUs.

## Training

VGAF trains five seeds (42–46) and reports on the official validation set.
GECV runs stratified 10-fold cross-validation.

```bash
# main model (soft hypergraph), 3-branch
python scripts/train_vgaf.py
python scripts/train_gecv.py

# a specific variant
python scripts/train_vgaf.py --ablation gat --output_name vgaf_gat

# 4-branch: stack the variant's _temporal ablation file
python scripts/train_vgaf.py --ablation mean_pool_temporal \
    --output_name vgaf_mean_pool_temporal

# the whole matrix
python scripts/run_ablations.py
```

Variants: `no_pose`, `mean_pool`, `gat`, `holistic` (no flag = hypergraph), each
with a `_temporal` counterpart for the 4-branch configuration, plus
`visual_only`, `audio_only`, `scene_only`, `temporal_only`.

Outputs land in `results/<run_name>/seed_<s>/` or `fold_<k>/`: `config.yaml`
(the fully resolved config), `metrics.json`, `history.json`, `best_model.pt`,
and `summary.json` at the run root.

## Decomposition ablations

These isolate why structured aggregators beat holistic pooling on VGAF. They use
`scripts/train_vgaf_b.py`, which differs from `train_vgaf.py` by one import line
(`diff` the two files) and builds `GERClassifierB`. For every published variant,
`GERClassifierB` delegates to `GERClassifier`, so initialisation is bit-identical:

```bash
python -m models.ger_classifier_b   # prints "identical" for all published cells
```

Two new configs: `mean_pool_uniform` replaces the learned temporal attention of
Eq. (6) with a uniform mean over valid frames; `mean_pool_nomlp_uniform` also
removes the residual refinement MLP, making it parameter-matched to Holistic.

```bash
# 3-branch
python scripts/train_vgaf_b.py --ablation mean_pool_uniform \
    --gpu 0 --output_name vgaf_mean_pool_uniform_3br
python scripts/train_vgaf_b.py --ablation mean_pool_nomlp_uniform \
    --gpu 1 --output_name vgaf_mean_pool_nomlp_uniform_3br

# 4-branch: mean_pool_temporal FIRST, then the decomposition file
python scripts/train_vgaf_b.py --ablation mean_pool_temporal --ablation mean_pool_uniform \
    --gpu 0 --output_name vgaf_mean_pool_uniform_4br
python scripts/train_vgaf_b.py --ablation mean_pool_temporal --ablation mean_pool_nomlp_uniform \
    --gpu 1 --output_name vgaf_mean_pool_nomlp_uniform_4br
```

**Ablation files merge left to right, so order matters.** `mean_pool_temporal`
also sets `backbone: mean_pool`; if it comes last it silently overrides
`mean_pool_nomlp` and trains the wrong model. Check the parameter count the
training log prints at the first seed: 1.52M for `mean_pool_uniform`, 1.38M for
`mean_pool_nomlp_uniform` under 4 branches.

To confirm a new run is comparable to a published one, diff the resolved configs:

```bash
python scripts/check_config_parity.py \
    --published results/vgaf_mean_pool_temporal/seed_42/config.yaml \
    --dataset vgaf --ablation mean_pool_temporal --ablation mean_pool_nomlp_uniform
```

It exits non-zero on any difference beyond the ablation keys.

## Evaluation and transfer

`scripts/evaluate.py` reproduces the reported numbers from checkpoints without
retraining:

```bash
# in-distribution (Table 1)
python scripts/evaluate.py --run_dir results/vgaf_gat --eval_dataset vgaf --ablation gat

# forward transfer VGAF -> GECV (Table 3)
python scripts/evaluate.py --run_dir results/vgaf_scene_only --eval_dataset gecv \
    --ablation scene_only --transfer

# reverse transfer GECV -> VGAF (Table 3)
python scripts/evaluate.py --run_dir results/gecv_no_pose --eval_dataset vgaf \
    --ablation no_pose --transfer
```

`--ablation` must match how the checkpoint was trained, since the architecture
has to match the weights. The whole matrix runs via
`python scripts/run_evaluations.py`, writing to `evaluations/eval_<stamp>/`.

Face-count stratified transfer (the >=10-face bucket) comes from
`scripts/eval_cross_dataset.py` and `scripts/eval_reverse_transfer.py`, using
the statistics in `stats/`.

## Reproducing the tables and figures

| Output | Command |
|---|---|
| Table 1 (overall) | `python scripts/aggregate_results.py` |
| Table 1 (per-class) | `python scripts/aggregate_per_class.py` |
| Table 3 (transfer) | `python scripts/aggregate_cross_eval.py` |
| Statistical tests | `python scripts/paired_tests.py` |
| Fusion weights | `python scripts/extract_fusion_weights.py` (needs checkpoints) |
| Computational cost | `python scripts/measure_cost_b.py --gpu 0 --include_ablations` |
| Figure 2 | `python scripts/make_figures_sbn.py` |
| Confusion matrices | `python scripts/aggregate_confusion.py` |

These run against the reference JSON files in `results/`, so most work without
retraining. Run the cost measurement on an idle GPU.

## Notes on the protocol

**GECV checkpoint selection.** Following Wang et al. (ICPR 2024), the held-out
fold is used both for early stopping and for reporting. GECV accuracies are
therefore optimistic relative to a nested protocol, and the GECV in-distribution
ceiling used to normalise transfer loss inherits that bias. VGAF figures are
unaffected: VGAF runs early-stop on a development split carved from the training
set, and the 766-clip validation set is used only for final reporting.

**Standard deviations.** The paper reports the *sample* standard deviation
(`ddof=1`). `summary.json` stores `np.std`, the population version (`ddof=0`).
For five seeds the difference is about 11%. The aggregation scripts report the
paper's convention.

**Reverse-transfer variance.** The 10 GECV folds share roughly 90% of their
training data, so per-fold standard deviations understate true variance and are
reported descriptively only.

**Hyperparameter tuning.** Learning rate, branch dropout and model width were
chosen on a separate stratified 20% split of the VGAF training set, disjoint
from the validation set, and applied to GECV without further tuning.
`configs/vgaf_tune.yaml` is the tuning configuration; the sweep driver script
and its outputs were not retained, so the search itself is documented here
rather than reproducible.

## Citation

Will be provided after final publication.


## License

MIT, see `LICENSE`. This covers the code in this repository only. The datasets
and pretrained backbones are governed by their own licences and terms.

## Contact

