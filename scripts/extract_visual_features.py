#!/usr/bin/env python3
"""
Unified per-clip visual feature extraction for VGAF Protocol A and GECV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm


# ── Locate project root ──────────────────────────────────────────────
def _find_project_root() -> Path:
    anchor = Path(__file__).resolve().parent
    for candidate in [anchor] + list(anchor.parents):
        if (candidate / "configs").is_dir() and (candidate / "requirements.txt").is_file():
            return candidate
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    raise FileNotFoundError("Cannot locate project root.")

_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing.face_detection import FaceDetector
from data.preprocessing.feature_extraction import PersonFeatureExtractor
from data.preprocessing.tracking import PersonTracker
from data.preprocessing.pose_extractor import PoseExtractor, OUT_POSE_DIM
from data.extract.discover import discover_clips
from utils.seed import set_seed

logger = logging.getLogger(__name__)

NUM_FRAMES = 15                          # T — fixed, matches Wang ICPR 2024
GECV_CLASSES = ("positive", "negative", "neutral")



def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str) -> dict[str, Any]:
    default_path = _PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)
    with open(config_path) as f:
        overrides = yaml.safe_load(f)
    return _deep_merge(cfg, overrides)



def sample_uniform_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> Tuple[list[np.ndarray], list[int], float, int, int]:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in {video_path}")

    indices = np.linspace(0, total - 1, num=num_frames).round().astype(int).tolist()
    frames: list[np.ndarray] = []
    last_good: Optional[np.ndarray] = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            last_good = frame
        else:
            # Repeat last good frame; if none yet, use a black frame.
            if last_good is not None:
                frames.append(last_good.copy())
            else:
                frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    cap.release()
    assert len(frames) == num_frames
    return frames, indices, native_fps, width, height




def process_single_clip(
    clip_info: dict[str, Any],
    detector: FaceDetector,
    tracker: Optional[PersonTracker],
    extractor: PersonFeatureExtractor,
    pose_extractor: PoseExtractor,
    cfg: dict[str, Any],
    output_dir: Path,
) -> Optional[str]:
    
    clip_id = clip_info["clip_id"]
    video_path = clip_info["video_path"]
    metadata = clip_info["metadata"]

    data_cfg = cfg["data"]
    max_persons = data_cfg["max_persons"]
    face_size = data_cfg["face_size"]
    body_size = data_cfg["body_size"]

    output_path = output_dir / f"{clip_id}.h5"

    try:
        # 1. Uniform T=15 frame sampling.
        frames, frame_indices, native_fps, img_w, img_h = sample_uniform_frames(
            video_path, num_frames=NUM_FRAMES,
        )
        T = NUM_FRAMES

        # 2. Face detection per frame.
        detections_per_frame = [detector.detect_faces(f) for f in frames]
        total_detections = sum(len(d) for d in detections_per_frame)

        if total_detections == 0:
            logger.warning("Clip %s: no faces detected — saving empty.", clip_id)
            _save_empty(
                output_path, T, max_persons,
                extractor.total_dim + OUT_POSE_DIM,
                frame_indices, metadata, native_fps, img_w, img_h,
            )
            return str(output_path)

        # 3. (Optional) tracking. When tracker is None, use raw detections.
        # GECV-style dense crowds break UBAM's tracker (IoU links fail across
        # T=15 sparse sampling; n_init=3 confirmation discards ~97% of dets).
        # Our hypergraph doesn't use cross-frame person identity, so the
        # tracker is contraindicated by default.
        if tracker is not None:
            tracking_result = tracker.process_video(detections_per_frame, frames=frames)
            frame_assignments = tracking_result["frame_assignments"]
            num_tracks_for_meta = len(tracking_result["tracks"])
        else:
            # Wrap each frame's detections in the existing (id, det) shape.
            # The id is just an enumeration index per frame — meaningless across
            # frames, but downstream code only iterates, never compares it.
            frame_assignments = [
                [(i, det) for i, det in enumerate(dets)]
                for dets in detections_per_frame
            ]
            num_tracks_for_meta = 0

        # 4. Visual feature extraction (face + body + bbox) — UBAM's batched flow.
        visual_dim = extractor.total_dim                                       # 1984
        full_dim = visual_dim + OUT_POSE_DIM                                   # 2048

        features = np.zeros((T, max_persons, full_dim), dtype=np.float16)
        person_mask = np.zeros((T, max_persons), dtype=bool)
        num_persons_per_frame = np.zeros(T, dtype=np.int32)
        raw_bboxes = np.zeros((T, max_persons, 4), dtype=np.float32)

        # Phase A: collect crops + per-frame slot→det mapping.
        all_face_tensors: list[torch.Tensor] = []
        all_body_tensors: list[torch.Tensor] = []
        all_bboxes_px: list[list[float]] = []
        crop_idx: list[tuple[int, int]] = []                                   # (t, slot)
        bbox_per_frame: list[np.ndarray] = []                                  # [t] -> [N_t, 4] px

        for t in range(T):
            assignments = frame_assignments[t]
            if not assignments:
                bbox_per_frame.append(np.zeros((0, 4), dtype=np.float32))
                continue
            if len(assignments) > max_persons:
                assignments = sorted(
                    assignments, key=lambda x: x[1].get("confidence", 0.0), reverse=True
                )[:max_persons]
                assignments.sort(key=lambda x: x[0])

            num_persons_per_frame[t] = len(assignments)

            this_frame_bboxes: list[list[float]] = []
            for slot, (_, det) in enumerate(assignments):
                bbox = det["bbox"]                                             # [x1,y1,x2,y2] px
                face_crop = FaceDetector.crop_face(frames[t], bbox, target_size=face_size)
                body_crop = FaceDetector.crop_upper_body(frames[t], bbox, target_size=body_size)

                all_face_tensors.append(extractor.preprocess_face_crop(face_crop))
                all_body_tensors.append(extractor.preprocess_body_crop(body_crop))
                all_bboxes_px.append(list(bbox))
                crop_idx.append((t, slot))
                this_frame_bboxes.append(list(bbox))

                # Save normalised bbox.
                raw_bboxes[t, slot] = np.array([
                    bbox[0] / max(img_w, 1),
                    bbox[1] / max(img_h, 1),
                    bbox[2] / max(img_w, 1),
                    bbox[3] / max(img_h, 1),
                ], dtype=np.float32)
                person_mask[t, slot] = True

            bbox_per_frame.append(np.asarray(this_frame_bboxes, dtype=np.float32))

        # Phase B: batched face/body/bbox feature extraction (UBAM pattern).
        all_face = torch.stack(all_face_tensors, dim=0)
        all_body = torch.stack(all_body_tensors, dim=0)
        all_bbox = torch.tensor(all_bboxes_px, dtype=torch.float32)
        batch_size = 64
        feat_chunks: list[torch.Tensor] = []
        for i in range(0, all_face.shape[0], batch_size):
            j = min(i + batch_size, all_face.shape[0])
            feat_chunks.append(
                extractor.extract_all(
                    all_face[i:j], all_body[i:j], all_bbox[i:j], img_w, img_h,
                ).cpu().float()
            )
        all_visual = torch.cat(feat_chunks, dim=0).numpy().astype(np.float32)  # [N_total, 1984]

        # Phase C: pose features per frame (uses canonical per-frame bboxes).
        # YOLOv8-pose ingests RGB frames; convert from BGR once.
        pose_buffer = np.zeros((T, max_persons, OUT_POSE_DIM), dtype=np.float32)
        for t in range(T):
            n_t = bbox_per_frame[t].shape[0]
            if n_t == 0:
                continue
            frame_rgb = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
            pose_frame = pose_extractor.extract_for_frame(
                frame_rgb, bbox_per_frame[t]
            )                                                                  # [n_t, 64]
            pose_buffer[t, :n_t] = pose_frame.astype(np.float32)

        # Phase D: scatter visual + pose into [T, N_max, 2048].
        for idx_, (t, slot) in enumerate(crop_idx):
            visual_vec = all_visual[idx_]                                      # [1984]
            pose_vec = pose_buffer[t, slot]                                    # [64]
            features[t, slot, :] = np.concatenate([visual_vec, pose_vec]).astype(np.float16)

        # 5. Save.
        meta_dict = {
            **metadata,
            "clip_id": clip_id,
            "video_path": video_path,
            "native_fps": native_fps,
            "num_frames": T,
            "max_persons": max_persons,
            "feature_dim": full_dim,
            "face_dim": 1408,
            "body_dim": 512,
            "bbox_dim": 64,
            "pose_dim": OUT_POSE_DIM,
            "img_width": img_w,
            "img_height": img_h,
            "total_detections": total_detections,
            "num_tracks": num_tracks_for_meta,
            "tracker_used": tracker is not None,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(output_path), "w") as hf:
            hf.create_dataset("features", data=features, compression="gzip", compression_opts=4)
            hf.create_dataset("person_mask", data=person_mask)
            hf.create_dataset("raw_bboxes", data=raw_bboxes)
            hf.create_dataset("num_persons", data=num_persons_per_frame)
            hf.create_dataset("frame_indices",
                              data=np.array(frame_indices, dtype=np.int32))
            hf.attrs["metadata"] = json.dumps(meta_dict)

        return str(output_path)

    except Exception as e:
        logger.error("Clip %s failed: %s", clip_id, e, exc_info=True)
        return None


def _save_empty(
    output_path: Path,
    T: int,
    max_persons: int,
    full_dim: int,
    frame_indices: list[int],
    metadata: dict,
    native_fps: float,
    img_w: int,
    img_h: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_dict = {
        **metadata,
        "native_fps": native_fps,
        "img_width": img_w,
        "img_height": img_h,
        "num_frames": T,
        "max_persons": max_persons,
        "feature_dim": full_dim,
        "total_detections": 0,
        "num_tracks": 0,
    }
    with h5py.File(str(output_path), "w") as hf:
        hf.create_dataset("features",
                          data=np.zeros((T, max_persons, full_dim), dtype=np.float16),
                          compression="gzip", compression_opts=4)
        hf.create_dataset("person_mask", data=np.zeros((T, max_persons), dtype=bool))
        hf.create_dataset("raw_bboxes", data=np.zeros((T, max_persons, 4), dtype=np.float32))
        hf.create_dataset("num_persons", data=np.zeros(T, dtype=np.int32))
        hf.create_dataset("frame_indices", data=np.array(frame_indices, dtype=np.int32))
        hf.attrs["metadata"] = json.dumps(meta_dict)




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-clip visual feature extraction (VGAF Protocol A + GECV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["vgaf", "gecv"], required=True)
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config providing data.* and model.backbone.* keys")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip clips whose HDF5 already exists.")
    parser.add_argument("--max_clips", type=int, default=0,
                        help="Process at most this many clips (0 = all).")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only shard K of N, e.g. '0/2' or '1/2'. "
                             "Clips are split by (idx %% N) so workload balances "
                             "across shards even with varied clip durations.")
    parser.add_argument("--yolo_model", type=str, default="yolov8x-pose.pt")
    parser.add_argument("--use_tracker", action="store_true",
                        help="Use UBAM's PersonTracker. Default OFF because it "
                             "rejects ~97%% of detections in dense crowds. Our "
                             "hypergraph doesn't use cross-frame person identity "
                             "so the tracker provides no benefit.")
    parser.add_argument("--split", default="all", choices=["train", "val", "all"],
        help="VGAF split. Ignored for GECV.",
    )
    parser.add_argument("--video_extension", type=str, default=".mp4")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    device_str = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    logger.info("=" * 60)
    logger.info("Visual Feature Extraction")
    logger.info("=" * 60)
    logger.info("  Dataset:     %s", args.dataset)
    logger.info("  Video root:  %s", args.video_root)
    logger.info("  Output:      %s", args.output_dir)
    logger.info("  Device:      %s", device_str)
    logger.info("  T (frames):  %d (uniform sampling)", NUM_FRAMES)
    logger.info("  YOLO model:  %s", args.yolo_model)
    logger.info("  Tracker:     %s", "ON" if args.use_tracker else "OFF (default)")

    # clips = discover_clips(args.dataset, args.video_root, args.video_extension)
    clips = discover_clips(args.dataset, Path(args.video_root), split=args.split)
    if not clips:
        logger.error("No clips found.")
        sys.exit(1)

    if args.max_clips > 0:
        clips = clips[: args.max_clips]
        logger.info("Limiting to %d clips", len(clips))

    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n and n >= 1):
            raise ValueError(f"Bad --shard {args.shard!r}; need K/N with 0<=K<N.")
        clips = [c for i, c in enumerate(clips) if i % n == k]
        logger.info("Shard %d/%d: %d clips on this worker", k, n, len(clips))

    if args.resume:
        before = len(clips)
        clips = [c for c in clips if not (args.output_dir / f"{c['clip_id']}.h5").exists()]
        logger.info("Resume: skipping %d already-processed.", before - len(clips))

    if not clips:
        logger.info("All clips already processed. Nothing to do.")
        return

    logger.info("Initialising face detector …")
    detector = FaceDetector(device=device_str)

    if args.use_tracker:
        logger.info("Initialising person tracker …")
        tracker = PersonTracker(max_age=30, n_init=3, max_iou_distance=0.7)
    else:
        logger.info("Person tracker DISABLED (pass --use_tracker to enable).")
        tracker = None

    logger.info("Initialising visual feature extractor …")
    face_weights = str(_PROJECT_ROOT / "weights" /
                       f"{cfg['model']['backbone']['face_model']}.pt")
    clip_weights = str(_PROJECT_ROOT / "weights" / "open_clip_pytorch_model.bin")
    extractor = PersonFeatureExtractor(
        device=device_str,
        face_model_name=cfg["model"]["backbone"]["face_model"],
        face_weights_path=face_weights,
        clip_model_name=cfg["model"]["backbone"]["body_model"],
        clip_weights_path=clip_weights,
        face_size=cfg["data"]["face_size"],
        body_size=cfg["data"]["body_size"],
    )

    logger.info("Initialising pose extractor …")
    pose_extractor = PoseExtractor(device=device_str, yolo_model=args.yolo_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    t0 = time.time()

    for clip_info in tqdm(clips, desc=f"Extracting [{args.dataset}]"):
        result = process_single_clip(
            clip_info=clip_info,
            detector=detector,
            tracker=tracker,
            extractor=extractor,
            pose_extractor=pose_extractor,
            cfg=cfg,
            output_dir=args.output_dir,
        )
        if result is not None:
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done.  ok=%d  failed=%d  elapsed=%.1fs (%.2fs/clip)",
                ok, fail, elapsed, elapsed / max(1, ok + fail))
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()