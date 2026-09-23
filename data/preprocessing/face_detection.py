"""
MTCNN-based face detection with confidence thresholding and crop utilities.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)


class FaceDetector:

    def __init__(
        self,
        device: str = "cuda",
        min_face_size: int = 20,
        thresholds: Optional[list[float]] = None,
        confidence_threshold: float = 0.9,
    ) -> None:
        
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        self.device = device
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold

        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=0.709,
            post_process=False,
            select_largest=False,
            keep_all=True,
            device=device,
        )



    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        
        # MTCNN expects RGB PIL-like input; convert from BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # facenet_pytorch MTCNN returns (boxes, probs, landmarks) or Nones
        boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)

        detections: list[dict] = []

        if boxes is None or probs is None:
            return detections

        h, w = frame.shape[:2]

        for i in range(len(boxes)):
            conf = float(probs[i])
            if conf < self.confidence_threshold:
                continue

            bbox = boxes[i].tolist()  # [x1, y1, x2, y2]

            # Clamp to image boundaries
            bbox = [
                max(0.0, bbox[0]),
                max(0.0, bbox[1]),
                min(float(w), bbox[2]),
                min(float(h), bbox[3]),
            ]

            # Skip degenerate boxes
            if bbox[2] - bbox[0] < 1.0 or bbox[3] - bbox[1] < 1.0:
                continue

            lm = None
            if landmarks is not None and landmarks[i] is not None:
                lm = landmarks[i].tolist()  # [[x, y], …] length 5

            detections.append(
                {
                    "bbox": bbox,
                    "confidence": conf,
                    "landmarks": lm,
                }
            )

        return detections

    

    @staticmethod
    def crop_face(
        frame: np.ndarray,
        bbox: list[float],
        target_size: int = 260,
        margin_ratio: float = 0.4,
    ) -> np.ndarray:
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        margin_x = bw * margin_ratio
        margin_y = bh * margin_ratio

        # Expand with margin, clamp to image
        cx1 = int(max(0, x1 - margin_x))
        cy1 = int(max(0, y1 - margin_y))
        cx2 = int(min(w, x2 + margin_x))
        cy2 = int(min(h, y2 + margin_y))

        # Guard against zero-area crops
        if cx2 <= cx1 or cy2 <= cy1:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        crop = frame[cy1:cy2, cx1:cx2]
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def crop_upper_body(
        frame: np.ndarray,
        bbox: list[float],
        target_size: int = 224,
        body_height_factor: float = 2.5,
    ) -> np.ndarray:
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) / 2.0

        # Body region spans wider and taller than the face
        body_w = face_w * 1.8
        body_h = face_h * body_height_factor

        bx1 = int(max(0, face_cx - body_w / 2.0))
        by1 = int(max(0, y1))  # Start from top of face
        bx2 = int(min(w, face_cx + body_w / 2.0))
        by2 = int(min(h, y1 + body_h))

        # Guard against zero-area crops
        if bx2 <= bx1 or by2 <= by1:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        crop = frame[by1:by2, bx1:bx2]
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized

    
    # Batch convenience
    def detect_and_crop(
        self,
        frame: np.ndarray,
        face_size: int = 260,
        body_size: int = 224,
    ) -> list[dict]:
        
        detections = self.detect_faces(frame)

        for det in detections:
            det["face_crop"] = self.crop_face(frame, det["bbox"], target_size=face_size)
            det["body_crop"] = self.crop_upper_body(frame, det["bbox"], target_size=body_size)

        return detections
