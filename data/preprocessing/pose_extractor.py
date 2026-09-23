"""
YOLOv8-pose feature extractor.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

POSE_PROJECTION_SEED = 42
RAW_POSE_DIM = 17 * 3   # COCO keypoints × (x, y, conf) = 51
OUT_POSE_DIM = 64
IOU_MATCH_THRESHOLD = 0.3


def make_pose_projection(seed: int = POSE_PROJECTION_SEED) -> nn.Linear:
    proj = nn.Linear(RAW_POSE_DIM, OUT_POSE_DIM, bias=False)
    g = torch.Generator().manual_seed(seed)
    W = torch.empty(OUT_POSE_DIM, RAW_POSE_DIM)
    try:
        nn.init.orthogonal_(W, generator=g)
    except TypeError:
        # Older torch lacks the generator kwarg — fall back to seeding globally.
        torch.manual_seed(seed)
        nn.init.orthogonal_(W)
    with torch.no_grad():
        proj.weight.copy_(W)
    for p in proj.parameters():
        p.requires_grad = False
    proj.eval()
    return proj


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """IoU between two sets of axis-aligned boxes. a [m,4], b [n,4] → [m,n]."""
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]))
    a = a_xyxy[:, None, :]
    b = b_xyxy[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def greedy_iou_match(
    canonical_xyxy: np.ndarray,
    yolo_xyxy: np.ndarray,
    threshold: float = IOU_MATCH_THRESHOLD,
) -> np.ndarray:
    """Greedy IoU match. Returns [N_canon] indices into yolo (-1 = no match)."""
    N = canonical_xyxy.shape[0]
    assigned = -np.ones(N, dtype=np.int64)
    if N == 0 or yolo_xyxy.shape[0] == 0:
        return assigned

    iou = iou_matrix(canonical_xyxy, yolo_xyxy)
    used_c = np.zeros(N, dtype=bool)
    used_y = np.zeros(yolo_xyxy.shape[0], dtype=bool)

    while True:
        masked = iou.copy()
        masked[used_c, :] = -1.0
        masked[:, used_y] = -1.0
        if masked.max() < threshold:
            break
        c, y = np.unravel_index(np.argmax(masked), masked.shape)
        assigned[c] = y
        used_c[c] = True
        used_y[y] = True
    return assigned


def normalise_keypoints(kps_pixel: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    """Convert pixel keypoints [17, 3] → bbox-relative [51] vector.

    Output layout: ``[x_rel_0, y_rel_0, conf_0, ..., x_rel_16, y_rel_16, conf_16]``.
    (x_rel, y_rel) may fall slightly outside [0, 1] when a keypoint sits just
    past the bbox edge (typical for head-tops); we don't clip.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    x_rel = (kps_pixel[:, 0] - x1) / w
    y_rel = (kps_pixel[:, 1] - y1) / h
    conf = kps_pixel[:, 2]
    return np.stack([x_rel, y_rel, conf], axis=-1).reshape(-1).astype(np.float32)


class PoseExtractor:
    
    def __init__(
        self,
        device: str = "cuda",
        yolo_model: str = "yolov8x-pose.pt",
        iou_threshold: float = IOU_MATCH_THRESHOLD,
        projection_seed: int = POSE_PROJECTION_SEED,
        yolo_imgsz: int = 640,
        yolo_conf: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        self.device = device
        self.iou_threshold = iou_threshold
        self.out_dim = OUT_POSE_DIM
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf

        logger.info("Loading YOLOv8-pose: %s", yolo_model)
        self.detector = YOLO(yolo_model)
        self.detector.to(device)
        self.projector = make_pose_projection(projection_seed).to("cpu")

    @torch.no_grad()
    def detect_frame(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        results = self.detector(
            frame_rgb,
            verbose=False,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
        )
        if not results:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 17, 3), dtype=np.float32)
        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None or len(res.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 17, 3), dtype=np.float32)
        bboxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        if res.keypoints is None or res.keypoints.data is None:
            return bboxes, np.zeros((bboxes.shape[0], 17, 3), dtype=np.float32)
        return bboxes, res.keypoints.data.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_for_frame(
        self,
        frame_rgb: np.ndarray,
        canonical_bboxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        
        N = canonical_bboxes_xyxy.shape[0]
        out = np.zeros((N, self.out_dim), dtype=np.float32)
        if N == 0:
            return out

        yolo_xyxy, yolo_kps = self.detect_frame(frame_rgb)
        assign = greedy_iou_match(
            canonical_bboxes_xyxy, yolo_xyxy, threshold=self.iou_threshold
        )

        matched = np.where(assign >= 0)[0]
        if matched.size == 0:
            return out

        raw_batch = np.stack(
            [
                normalise_keypoints(yolo_kps[assign[i]], canonical_bboxes_xyxy[i])
                for i in matched
            ],
            axis=0,
        )                                                                       # [n_matched, 51]
        proj_batch = self.projector(torch.from_numpy(raw_batch)).numpy()        # [n_matched, 64]
        out[matched] = proj_batch.astype(np.float32)
        return out