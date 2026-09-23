"""
VGAF dataset loader.
Loads pre-extracted multimodal features for the original (un-merged)
5-second VGAF clips. Three classes with the canonical VGAF/UBAM index
ordering::
    positive = 0
    neutral  = 1
    negative = 2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Class ordering — MUST match gecv_dataset.py after the flip.
# ----------------------------------------------------------------------
CLASS_NAMES: Tuple[str, str, str] = ("positive", "neutral", "negative")
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

# ----------------------------------------------------------------------
# Raw VGAF label int → class name.
#
# Verified against VGAF_readme.txt:
#     1 = Positive, 2 = Neutral, 3 = Negative
# i.e. raw_label - 1 == CLASS_TO_IDX[name] under our (positive, neutral,
# negative) ordering. If you ever swap CLASS_NAMES, update this dict too.
# ----------------------------------------------------------------------
RAW_LABEL_TO_NAME: Dict[int, str] = {
    1: "positive",
    2: "neutral",
    3: "negative",
}

T_FRAMES = 15
N_MAX = 32  # canonical persons per clip — must match the extractor.


class VGAFDataset(Dataset):

    def __init__(
        self,
        video_root: str,
        cache_root: str,
        split: str = "train",
        strict: bool = True,
        use_temporal: bool = False, 
    ):
        self.use_temporal = use_temporal
        self.video_root = Path(video_root)
        self.cache_root = Path(cache_root)
        self.split = split.lower()
        if self.split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        labels_file = self.video_root / f"{self.split.capitalize()}_labels.txt"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        self._all_records: List[Tuple[str, int]] = []  # (clip_id, label_idx)
        with open(labels_file) as f:
            header = f.readline().strip().split()
            if header[:2] != ["Vid_name", "Label"]:
                raise ValueError(
                    f"Unexpected labels header in {labels_file}: {header}. "
                    f"Expected 'Vid_name Label'."
                )
            for line_no, line in enumerate(f, start=2):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 2:
                    raise ValueError(
                        f"Malformed line {line_no} in {labels_file}: {line!r}"
                    )
                vid_name, raw_label_str = parts[0], parts[1]
                try:
                    raw_label = int(raw_label_str)
                except ValueError as e:
                    raise ValueError(
                        f"Non-integer label at line {line_no} in {labels_file}: "
                        f"{raw_label_str!r}"
                    ) from e
                if raw_label not in RAW_LABEL_TO_NAME:
                    raise ValueError(
                        f"Unknown raw label {raw_label} at line {line_no} in "
                        f"{labels_file}. Update RAW_LABEL_TO_NAME or check "
                        f"VGAF_readme.txt."
                    )
                class_name = RAW_LABEL_TO_NAME[raw_label]
                label_idx = CLASS_TO_IDX[class_name]
                clip_id = f"{self.split}_{vid_name}"
                self._all_records.append((clip_id, label_idx))

        # Filter to clips with extracted features.
        present: List[Tuple[str, int]] = []
        missing: List[str] = []
        for clip_id, lbl in self._all_records:
            if self._has_features(clip_id):
                present.append((clip_id, lbl))
            else:
                missing.append(clip_id)

        if strict and missing:
            raise FileNotFoundError(
                f"{len(missing)} clips missing features under {self.cache_root}. "
                f"First few: {missing[:5]}. Run the extractors or pass "
                f"strict=False to drop them silently."
            )
        self.records = present

        if not self.records:
            raise RuntimeError(
                f"No clips with features for split={split} under {self.cache_root}."
            )

        logger.info(
            "VGAF %s: %d clips in labels file, %d with features (%d missing).",
            split,
            len(self._all_records),
            len(self.records),
            len(missing),
        )

    
    def _has_features(self, clip_id: str) -> bool:
        subs = ["visual", "audio", "scene"]
        if self.use_temporal:
            subs.append("temporal")
        for sub in subs:
            if not (self.cache_root / sub / f"{clip_id}.h5").exists():
                return False
        return True

    
    @property
    def all_clip_ids(self) -> List[str]:
        return [c for c, _ in self.records]

    @property
    def all_labels(self) -> List[int]:
        return [l for _, l in self.records]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for CE loss."""
        counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for _, lbl in self.records:
            counts[lbl] += 1
        weights = counts.sum() / (len(CLASS_NAMES) * counts.clip(min=1))
        return torch.tensor(weights, dtype=torch.float32)

    
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        clip_id, label = self.records[idx]
        return self._load_clip(clip_id, label)

    def _load_clip(self, clip_id: str, label: int) -> Dict:

        with h5py.File(self.cache_root / "visual" / f"{clip_id}.h5", "r") as f:
            visual = torch.from_numpy(f["features"][...]).float()       # [T, N_max, 2048]
            person_mask = torch.from_numpy(f["person_mask"][...]).bool()  # [T, N_max]

        with h5py.File(self.cache_root / "audio" / f"{clip_id}.h5", "r") as f:
            audio = torch.from_numpy(f["features"][...]).float()         # [1024]

        with h5py.File(self.cache_root / "scene" / f"{clip_id}.h5", "r") as f:
            scene = torch.from_numpy(f["features"][...]).float()         # [512]
        

        out = {
            "clip_id": clip_id,
            "label": torch.tensor(label, dtype=torch.long),
            "per_person_features": visual,
            "person_mask": person_mask,
            "audio": audio,
            "scene": scene,
        }
        if self.use_temporal:
            with h5py.File(self.cache_root / "temporal" / f"{clip_id}.h5", "r") as f:
                out["temporal"] = torch.from_numpy(f["features"][...]).float()  # [768]
        
        return out



if __name__ == "__main__":
    
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    video_root = os.environ.get(
        "VGAF_VIDEO_ROOT", "/path/to/datasets/VGAFDATA/VGAF"
    )
    cache_root = os.environ.get(
        "VGAF_CACHE_ROOT",
        "/path/to/datasets/gerhyperg_features/vgaf",
    )

    print(f"Video root:  {video_root}")
    print(f"Cache root:  {cache_root}")
    print(f"Class order: {CLASS_NAMES}  (positive=0, neutral=1, negative=2)")
    print(f"Raw label → name: {RAW_LABEL_TO_NAME}")
    print()

    for split in ("train", "val"):
        try:
            ds = VGAFDataset(
                video_root=video_root,
                cache_root=cache_root,
                split=split,
                strict=False,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[{split}] ERROR: {e}")
            continue

        labels_all = [l for _, l in ds._all_records]
        print(f"[{split}] labels file: {len(ds._all_records)} clips")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name:8s} (idx {i}): {labels_all.count(i):4d}")
        print(f"    with features: {len(ds)} / {len(ds._all_records)}")
        if len(ds) > 0:
            weights = ds.class_weights().tolist()
            print("Class weights (CE):")
            for name, w in zip(CLASS_NAMES, weights):
                print(f"  {name:8s}: {w:.4f}")
            # print(f"    class weights (CE): {ds.class_weights().tolist()}")
        
        print()