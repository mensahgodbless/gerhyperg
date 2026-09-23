"""
Run the ablation matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for c in [here] + list(here.parents):
        if (c / "configs").is_dir() and (c / "requirements.txt").is_file():
            return c
    raise FileNotFoundError("project root not found")


_PROJECT_ROOT = _find_project_root()
logger = logging.getLogger(__name__)


DEFAULT_VGAF_ABLATIONS = [
    None,            # main run
    "no_pose",
    "mean_pool",
    "gat",
    "holistic",
    "visual_only",
    "audio_only",
    "scene_only",
]
DEFAULT_GECV_ABLATIONS = [
    None,            # main run
    "no_pose",
    "mean_pool",
]


def _run_train(script: str, ablation: Optional[str], output_name: str,
               log_path: Path, gpu: Optional[int] = None) -> int:
    cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / script),
           "--output_name", output_name]
    if ablation is not None:
        cmd.extend(["--ablation", ablation])
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])

    logger.info("$ %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def _read_summary(run_root: Path) -> dict:
    summary_path = run_root / "summary.json"
    if not summary_path.exists():
        return {"error": f"no summary.json at {summary_path}"}
    with open(summary_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="*", default=None,
                        help="override default ablation list")
    parser.add_argument("--vgaf_only", action="store_true")
    parser.add_argument("--gecv_only", action="store_true")
    parser.add_argument("--gecv_ablations", nargs="*", default=None,
                        help="override default GECV ablation subset")
    parser.add_argument("--output_root", type=Path,
                        default=_PROJECT_ROOT / "results")
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index passed through to each training run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    vgaf_list = args.ablations if args.ablations is not None else DEFAULT_VGAF_ABLATIONS
    # Allow "main" → None translation for CLI ergonomics.
    vgaf_list = [None if a in ("main", "none", "None") else a for a in vgaf_list]
    gecv_list = args.gecv_ablations if args.gecv_ablations is not None else DEFAULT_GECV_ABLATIONS
    gecv_list = [None if a in ("main", "none", "None") else a for a in gecv_list]

    matrix_root = args.output_root / f"ablations_{args.stamp}"
    matrix_root.mkdir(parents=True, exist_ok=True)
    logger.info("Matrix root: %s", matrix_root)

    results = {"vgaf": [], "gecv": []}

    if not args.gecv_only:
        for ab in vgaf_list:
            tag = ab or "main"
            name = f"vgaf_{tag}"
            run_dir = args.output_root / name
            log_path = matrix_root / f"{name}.log"
            logger.info("──[VGAF %s]────────────────────────────", tag)
            rc = _run_train("train_vgaf.py", ab, name, log_path, gpu=args.gpu)
            summary = _read_summary(run_dir)
            results["vgaf"].append({
                "ablation": tag, "returncode": rc,
                "run_dir": str(run_dir), "summary": summary,
            })

    if not args.vgaf_only:
        for ab in gecv_list:
            tag = ab or "main"
            name = f"gecv_{tag}"
            run_dir = args.output_root / name
            log_path = matrix_root / f"{name}.log"
            logger.info("──[GECV %s]────────────────────────────", tag)
            rc = _run_train("train_gecv.py", ab, name, log_path, gpu=args.gpu)
            summary = _read_summary(run_dir)
            results["gecv"].append({
                "ablation": tag, "returncode": rc,
                "run_dir": str(run_dir), "summary": summary,
            })

    with open(matrix_root / "matrix_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Matrix complete. Summary: %s", matrix_root / "matrix_summary.json")
    # Headline table.
    logger.info("\nVGAF (test acc on official Val):")
    for r in results["vgaf"]:
        s = r["summary"]
        if "test_acc_mean" in s:
            logger.info("  %-15s  %.4f ± %.4f", r["ablation"],
                        s["test_acc_mean"], s["test_acc_std"])
        else:
            logger.info("  %-15s  FAILED (%s)", r["ablation"], s.get("error", "rc=" + str(r["returncode"])))
    logger.info("\nGECV (val acc across folds):")
    for r in results["gecv"]:
        s = r["summary"]
        if "val_acc_mean" in s:
            logger.info("  %-15s  %.4f ± %.4f", r["ablation"],
                        s["val_acc_mean"], s["val_acc_std"])
        else:
            logger.info("  %-15s  FAILED", r["ablation"])


if __name__ == "__main__":
    main()
