"""
Evaluate every trained ablation with scripts/evaluate.py, into a separate dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import subprocess
import sys
import time
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for c in [here] + list(here.parents):
        if (c / "configs").is_dir() and (c / "requirements.txt").is_file():
            return c
    raise FileNotFoundError("project root not found")


_PROJECT_ROOT = _find_project_root()
logger = logging.getLogger(__name__)

# Match the training matrix in run_ablations.py.
DEFAULT_VGAF_ABLATIONS = [
    "main", "no_pose", "mean_pool", "gat", "holistic", "hypergraph_temporal", "no_pose_temporal", 
    "mean_pool_temporal", "gat_temporal", "holistic_temporal", "visual_only", "audio_only", "scene_only", "temporal_only"
]
DEFAULT_GECV_ABLATIONS = [
    "main", "no_pose", "mean_pool", "gat", "holistic", "hypergraph_temporal", "no_pose_temporal", 
    "mean_pool_temporal", "gat_temporal", "holistic_temporal", "visual_only", "audio_only", "scene_only", "temporal_only"
]


def _has_checkpoints(run_dir: Path) -> bool:
    return bool(glob.glob(str(run_dir / "seed_*" / "best_model.pt")) or
               glob.glob(str(run_dir / "fold_*" / "best_model.pt")))


def _ablation_args(ab: str):
    """evaluate.py takes no --ablation for the full (main) model."""
    return [] if ab in ("main", "none", "None") else ["--ablation", ab]


def _run_eval(run_dir, eval_dataset, ablation, transfer, out_json, log_path,
              gpu=None, dump_predictions=None) -> int:
    cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "evaluate.py"),
           "--run_dir", str(run_dir),
           "--eval_dataset", eval_dataset,
           "--out_json", str(out_json)]
    cmd += _ablation_args(ablation)
    if transfer:
        cmd.append("--transfer")
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if dump_predictions is not None:
        cmd += ["--dump_predictions", str(dump_predictions)]

    logger.info("$ %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=_PROJECT_ROOT / "results",
                    help="where training wrote run dirs (vgaf_*/, gecv_*/)")
    ap.add_argument("--output_root", type=Path, default=_PROJECT_ROOT / "evaluations",
                    help="where to write evaluation outputs (separate from results/)")
    ap.add_argument("--ablations", nargs="*", default=None,
                    help="VGAF-trained ablations to evaluate (default: full matrix)")
    ap.add_argument("--gecv_ablations", nargs="*", default=None,
                    help="GECV-trained ablations to evaluate (default: main no_pose mean_pool)")
    ap.add_argument("--mode", choices=["indist", "transfer", "all"], default="all",
                    help="in-distribution only, transfer only, or both")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index passed to each eval")
    ap.add_argument("--dump_predictions", action="store_true",
                    help="also write per-clip prediction CSVs into the output dir")
    ap.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    vgaf_abls = args.ablations if args.ablations is not None else DEFAULT_VGAF_ABLATIONS
    gecv_abls = args.gecv_ablations if args.gecv_ablations is not None else DEFAULT_GECV_ABLATIONS

    out_root = args.output_root / f"eval_{args.stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output root: %s", out_root)

    # Job list: (name, run_dir, eval_dataset, ablation, transfer).
    #   vgaf_<ab>  -> in-distribution VGAF, and forward transfer onto GECV
    #   gecv_<ab>  -> in-distribution GECV, and reverse transfer onto VGAF
    jobs = []
    for ab in vgaf_abls:
        rd = args.results_root / f"vgaf_{ab}"
        if args.mode in ("indist", "all"):
            jobs.append((f"vgaf_{ab}__indist", rd, "vgaf", ab, False))
        if args.mode in ("transfer", "all"):
            jobs.append((f"vgaf_{ab}__transfer_gecv", rd, "gecv", ab, True))
    for ab in gecv_abls:
        rd = args.results_root / f"gecv_{ab}"
        if args.mode in ("indist", "all"):
            jobs.append((f"gecv_{ab}__indist", rd, "gecv", ab, False))
        if args.mode in ("transfer", "all"):
            jobs.append((f"gecv_{ab}__transfer_vgaf", rd, "vgaf", ab, True))

    results, skipped = [], []
    for name, rd, eval_ds, ab, transfer in jobs:
        if not _has_checkpoints(rd):
            logger.warning("SKIP %s - no checkpoints under %s", name, rd)
            skipped.append({"name": name, "run_dir": str(rd), "reason": "no checkpoints"})
            continue
        out_json = out_root / f"{name}.json"
        log_path = out_root / f"{name}.log"
        dump = (out_root / f"{name}_preds.csv") if args.dump_predictions else None
        logger.info("-- %s --------------------------------", name)
        rc = _run_eval(rd, eval_ds, ab, transfer, out_json, log_path,
                       gpu=args.gpu, dump_predictions=dump)
        entry = {"name": name, "returncode": rc, "run_dir": str(rd),
                 "eval_dataset": eval_ds, "ablation": ab, "transfer": transfer}
        if out_json.exists():
            with open(out_json) as f:
                entry["metrics"] = json.load(f)
        else:
            logger.warning("  %s produced no JSON (rc=%s); see %s", name, rc, log_path)
        results.append(entry)

    summary = {"output_root": str(out_root), "results": results, "skipped": skipped}
    with open(out_root / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Headline table.
    logger.info("=" * 72)
    logger.info("EVALUATION SUMMARY   (accuracy %% +/- std  |  F1)")
    for key, title in [("indist", "In-distribution"),
                       ("transfer", "Cross-dataset transfer")]:
        rows = [r for r in results if key in r["name"]]
        if not rows:
            continue
        logger.info("")
        logger.info("%s:", title)
        for r in rows:
            m = r.get("metrics")
            if m:
                logger.info("  %-34s %6.2f +/- %4.2f   |  %-11s %5.2f",
                            r["name"], m["accuracy_mean"], m["accuracy_std"],
                            m["f1_name"], m["f1_mean"])
            else:
                logger.info("  %-34s FAILED (rc=%s)", r["name"], r["returncode"])
    if skipped:
        logger.info("")
        logger.info("Skipped (%d): %s", len(skipped),
                    ", ".join(s["name"] for s in skipped))
    logger.info("")
    logger.info("All outputs in %s", out_root)


if __name__ == "__main__":
    main()
