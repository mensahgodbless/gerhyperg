"""
DeepSORT-based multi-person tracking across video frames.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)


class PersonTracker:

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = 100,
        max_iou_distance: float = 0.7,
        embedder: Optional[str] = None,
        embedder_gpu: bool = True,
    ) -> None:
        
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self._has_embedder = embedder is not None

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
        )

    def reset(self) -> None:
        self.tracker.delete_all_tracks()

    
    # Core tracking
    def process_video(
        self,
        detections_per_frame: list[list[dict]],
        frames: Optional[list[np.ndarray]] = None,
    ) -> dict:
        
        self.reset()

        num_frames = len(detections_per_frame)
        tracks: dict[int, list[tuple[int, list[float], float]]] = {}
        frame_assignments: list[list[tuple[int, dict]]] = [[] for _ in range(num_frames)]

        for t in range(num_frames):
            dets = detections_per_frame[t]
            frame = frames[t] if frames is not None else None

            # Build DeepSort input: list of ([x1, y1, w, h], confidence, class)
            ds_detections: list[tuple[list[float], float, str]] = []
            det_lookup: list[dict] = []

            for det in dets:
                bbox = det["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                if w < 1.0 or h < 1.0:
                    continue
                conf = det["confidence"]
                # DeepSort expects ([left, top, w, h], confidence, detection_class)
                ds_detections.append(([x1, y1, w, h], conf, "person"))
                det_lookup.append(det)


            if self._has_embedder and frame is not None:
                raw_tracks = self.tracker.update_tracks(
                    ds_detections, frame=frame
                )
            else:
                n_dets = len(ds_detections)
                if n_dets > 0:
                    unit = np.ones(128, dtype=np.float32)
                    unit /= np.linalg.norm(unit)
                    dummy_embeds = [unit.copy() for _ in range(n_dets)]
                else:
                    dummy_embeds = []
                raw_tracks = self.tracker.update_tracks(
                    ds_detections, embeds=dummy_embeds
                )

            # Collect confirmed tracks for this frame
            for track in raw_tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                # DeepSort returns ltrb (left, top, right, bottom)
                ltrb = track.to_ltrb()
                bbox_out = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]

                # Find the detection closest to this track's bbox
                matched_det = self._match_track_to_detection(bbox_out, det_lookup)

                if track_id not in tracks:
                    tracks[track_id] = []

                det_conf = matched_det["confidence"] if matched_det is not None else 0.0
                tracks[track_id].append((t, bbox_out, det_conf))

                # Build frame assignment entry
                assignment_det = matched_det if matched_det is not None else {
                    "bbox": bbox_out,
                    "confidence": det_conf,
                    "landmarks": None,
                }
                # Preserve the track bbox for downstream (more temporally stable)
                assignment_det = dict(assignment_det)
                assignment_det["bbox"] = bbox_out
                frame_assignments[t].append((track_id, assignment_det))

            # Sort each frame's assignments by track_id for consistency
            frame_assignments[t].sort(key=lambda x: x[0])

        logger.info(
            "Tracking complete: %d tracks across %d frames", len(tracks), num_frames
        )

        return {
            "tracks": tracks,
            "frame_assignments": frame_assignments,
        }

   
    @staticmethod
    def _match_track_to_detection(
        track_bbox: list[float],
        detections: list[dict],
        iou_threshold: float = 0.3,
    ) -> Optional[dict]:
        
        if not detections:
            return None

        best_iou = 0.0
        best_det: Optional[dict] = None

        for det in detections:
            iou = PersonTracker._compute_iou(track_bbox, det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_det = det

        if best_iou >= iou_threshold:
            return best_det
        return None

    @staticmethod
    def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
        
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        if inter == 0.0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        if union <= 0.0:
            return 0.0

        return inter / union

    
    # Utilities
    @staticmethod
    def filter_short_tracks(
        tracks: dict[int, list[tuple[int, list[float], float]]],
        min_length: int = 3,
    ) -> dict[int, list[tuple[int, list[float], float]]]:
        
        return {
            tid: entries
            for tid, entries in tracks.items()
            if len(entries) >= min_length
        }

    @staticmethod
    def get_active_track_ids_at_frame(
        frame_assignments: list[list[tuple[int, dict]]],
        frame_idx: int,
    ) -> list[int]:
        
        if frame_idx < 0 or frame_idx >= len(frame_assignments):
            return []
        return [tid for tid, _ in frame_assignments[frame_idx]]

    @staticmethod
    def interpolate_missing_frames(
        tracks: dict[int, list[tuple[int, list[float], float]]],
        total_frames: int,
    ) -> dict[int, list[tuple[int, list[float], float]]]:
        
        interpolated: dict[int, list[tuple[int, list[float], float]]] = {}

        for tid, entries in tracks.items():
            if len(entries) < 2:
                interpolated[tid] = list(entries)
                continue

            entries_sorted = sorted(entries, key=lambda x: x[0])
            frame_to_entry = {e[0]: e for e in entries_sorted}

            first_frame = entries_sorted[0][0]
            last_frame = entries_sorted[-1][0]

            filled: list[tuple[int, list[float], float]] = []

            for f in range(first_frame, last_frame + 1):
                if f in frame_to_entry:
                    filled.append(frame_to_entry[f])
                else:
                    # Find surrounding observed frames
                    prev_f = max(pf for pf in frame_to_entry if pf < f)
                    next_f = min(nf for nf in frame_to_entry if nf > f)
                    prev_entry = frame_to_entry[prev_f]
                    next_entry = frame_to_entry[next_f]

                    alpha = (f - prev_f) / (next_f - prev_f)
                    interp_bbox = [
                        prev_entry[1][j] + alpha * (next_entry[1][j] - prev_entry[1][j])
                        for j in range(4)
                    ]
                    interp_conf = prev_entry[2] + alpha * (next_entry[2] - prev_entry[2])
                    filled.append((f, interp_bbox, interp_conf))

            interpolated[tid] = filled

        return interpolated