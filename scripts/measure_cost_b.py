"""
Computational cost of each visual aggregation variant. ADDITIVE.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
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


_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from training.train_utils import load_config  # noqa: E402  (untouched original)
from training.train_utils_b import build_model  # noqa: E402

# (label, ablation files) for the five published variants.
# "main" = no ablation file = hypergraph with pose.
PUBLISHED = [
    ("Hypergraph", []),
    ("No-pose", ["no_pose"]),
    ("Mean-pool", ["mean_pool"]),
    ("GAT", ["gat"]),
    ("Holistic", ["holistic"]),
]
DECOMPOSITION = [
    ("Mean-pool-uniform", ["mean_pool_uniform"]),
    ("Mean-pool-noMLP-uniform", ["mean_pool_nomlp_uniform"]),
]

VGAF_TRAIN_CLIPS = 2129  # inner-train size after the seed-7 dev carve


def _gpu_busy_warning(gpu: int) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            print(f"WARNING: other processes are using GPU {gpu}:\n{out}\n"
                  "Timings will be contaminated. Use an idle GPU.\n")
    except Exception:
        pass


def _inputs(batch, frames, persons, dim, device, temporal):
    x = torch.randn(batch, frames, persons, dim, device=device)
    mask = torch.ones(batch, frames, persons, dtype=torch.bool, device=device)
    audio = torch.randn(batch, 1024, device=device)
    scene = torch.randn(batch, 512, device=device)
    temp = torch.randn(batch, 768, device=device) if temporal else None
    return x, mask, audio, scene, temp


def _forward(model, inputs):
    x, mask, audio, scene, temp = inputs
    return model(x, mask, audio, scene, temporal=temp)


def _resolve_input_dim(model, frames, persons, device, temporal):
    """Published no_pose runs may slice internally or expect 1984-d input.
    Try the full 2048-d cache vector first, fall back to the projector dim."""
    model.eval()
    for dim in (2048, 1984):
        try:
            with torch.inference_mode():
                _forward(model, _inputs(1, frames, persons, dim, device, temporal))
            return dim
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("could not find a valid per-person input dim")


def _cuda_time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _gflops(model, inputs):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return float("nan")
    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.inference_mode(), counter:
        _forward(model, inputs)
    return counter.get_total_flops() / 1e9


def measure(label, ablations, temporal, args, device):
    cfg = load_config("vgaf", ablations, {})
    cfg = copy.deepcopy(cfg)
    cfg["model"].setdefault("temporal", {})["enabled"] = temporal

    torch.manual_seed(0)
    model = build_model(cfg).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_visual = sum(p.numel() for p in model.visual_branch.parameters()
                        if p.requires_grad)

    dim = _resolve_input_dim(model, args.frames, args.persons, device, temporal)

    # FLOPs, one clip, fp32.
    gflops = _gflops(model, _inputs(1, args.frames, args.persons, dim, device, temporal))

    # Inference latency, AMP, as in evaluation.
    model.eval()
    in1 = _inputs(1, args.frames, args.persons, dim, device, temporal)
    in32 = _inputs(args.batch, args.frames, args.persons, dim, device, temporal)

    def infer(inp):
        def f():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _forward(model, inp)
        return f

    infer_b1 = _cuda_time(infer(in1), args.warmup, args.iters)
    infer_b32 = _cuda_time(infer(in32), args.warmup, args.iters) / args.batch

    # Training step, AMP + GradScaler + AdamW, as in training.
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    y = torch.randint(0, 3, (args.batch,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = _forward(model, in32)
            loss = F.cross_entropy(out["logits"].float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    torch.cuda.reset_peak_memory_stats(device)
    train_ms = _cuda_time(step, args.warmup, args.iters)
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20

    steps_per_epoch = math.ceil(VGAF_TRAIN_CLIPS / args.batch)

    row = {
        "variant": label,
        "branches": 4 if temporal else 3,
        "input_dim": dim,
        "params_total": params_total,
        "params_visual": params_visual,
        "gflops_clip": round(gflops, 4),
        "infer_ms_b1": round(infer_b1, 3),
        "infer_ms_clip_b32": round(infer_b32, 4),
        "train_ms_step": round(train_ms, 3),
        "train_s_epoch": round(train_ms * steps_per_epoch / 1000, 3),
        "peak_mem_mb": round(peak_mb, 1),
    }
    del model, opt
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frames", type=int, default=15,
                    help="visual frames per clip (paper: 15)")
    ap.add_argument("--persons", type=int, default=20,
                    help="person slots per frame (VGAF N_max: 20)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--include_ablations", action="store_true",
                    help="also measure the two decomposition variants")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for timing.")
        return 1
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    _gpu_busy_warning(args.gpu)

    variants = PUBLISHED + (DECOMPOSITION if args.include_ablations else [])
    rows = []
    for temporal in (False, True):
        for label, abl in variants:
            print(f"measuring {label:26s} {'4' if temporal else '3'}-br ...",
                  flush=True)
            rows.append(measure(label, abl, temporal, args, device))

    hdr = (f"{'variant':26s} br  {'params':>9s} {'visual':>9s} {'GFLOPs':>7s} "
           f"{'ms@b1':>7s} {'ms/clip@32':>10s} {'ms/step':>8s} {'s/epoch':>8s} {'MB':>7s}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:26s} {r['branches']}  {r['params_total']:9d} "
              f"{r['params_visual']:9d} {r['gflops_clip']:7.3f} "
              f"{r['infer_ms_b1']:7.3f} {r['infer_ms_clip_b32']:10.4f} "
              f"{r['train_ms_step']:8.3f} {r['train_s_epoch']:8.3f} "
              f"{r['peak_mem_mb']:7.1f}")

    out = args.out or (_ROOT / "results" /
                       f"cost_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "frames": args.frames, "persons": args.persons,
        "batch": args.batch, "warmup": args.warmup, "iters": args.iters,
        "note": "trainable model on cached features only; frozen backbones "
                "identical across variants and excluded",
    }
    with open(out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())