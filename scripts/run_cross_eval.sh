#!/bin/bash
# Cross-evaluate every VGAF checkpoint on GECV.
# Outputs: results/vgaf_<ablation>/seed_<N>/cross_eval_gecv/report.json

set -e
cd "$(dirname "$0")/.."

GPU=${1:-3}

ABLATIONS=(main no_pose mean_pool gat holistic
           hypergraph_temporal no_pose_temporal mean_pool_temporal gat_temporal holistic_temporal
           temporal_only visual_only audio_only scene_only)

n_done=0
n_failed=0

for ab in "${ABLATIONS[@]}"; do
    for seed_dir in results/vgaf_${ab}/seed_*/; do
        [ -d "$seed_dir" ] || continue

        ckpt="${seed_dir}best_model.pt"
        if [ ! -f "$ckpt" ]; then
            echo "skip (no checkpoint): $seed_dir"
            continue
        fi

        out="${seed_dir}cross_eval_gecv"
        if [ -f "${out}/report.json" ]; then
            echo "skip (already done): $seed_dir"
            continue
        fi

        echo "==> ${ab}/$(basename ${seed_dir%/})"
        if python scripts/eval_cross_dataset.py \
            --checkpoint "$ckpt" \
            --config_dir "$seed_dir" \
            --target_dataset gecv \
            --gpu $GPU \
            >> logs/cross_eval_${ab}.log 2>&1; then
            n_done=$((n_done + 1))
        else
            echo "  FAILED — see logs/cross_eval_${ab}.log"
            n_failed=$((n_failed + 1))
        fi
    done
done

echo
echo "Done. completed=$n_done failed=$n_failed"
