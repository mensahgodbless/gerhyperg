"""
GECV dataset loader 
Loads pre-extracted multimodal features for the Group Emotion Composite Video
benchmark. Three classes (positive 201 / negative 100 / neutral 107) with
labels derived from directory structure. 
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES: Tuple[str, ...] = ("positive", "neutral", "negative")
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}


class GECVDataset(Dataset):
   
    def __init__(
        self,
        video_root: Union[Path, str],
        cache_root: Union[Path, str],
        clip_ids: Optional[List[str]] = None,
        per_person_dim: int = 2048,
        audio_dim: int = 1024,
        scene_dim: int = 512,
        temporal_dim: int = 768,   
        trim_to_active_slots: bool = True,
        strict: bool = True,
        video_extension: str = ".mp4",
        use_temporal: bool = False,
    ) -> None:
        super().__init__()
        self.video_root = Path(video_root)
        self.cache_root = Path(cache_root)
        self.per_person_dim = per_person_dim
        self.audio_dim = audio_dim
        self.scene_dim = scene_dim
        self.temporal_dim = temporal_dim  
        self.trim_to_active_slots = trim_to_active_slots
        self.strict = strict
        self.use_temporal = use_temporal 

        records: List[Tuple[str, int]] = []
        for cls in CLASS_NAMES:
            cls_dir = self.video_root / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class dir: {cls_dir}")
            for vp in sorted(cls_dir.glob(f"*{video_extension}")):
                records.append((f"{cls}_{vp.stem}", CLASS_TO_IDX[cls]))

        if clip_ids is not None:
            wanted = set(clip_ids)
            records = [r for r in records if r[0] in wanted]
            if len(records) != len(wanted):
                missing = wanted - {r[0] for r in records}
                raise ValueError(
                    f"Requested {len(wanted)} clip_ids but matched only "
                    f"{len(records)}. Missing (first 5): {sorted(missing)[:5]}"
                )

        self.records: List[Tuple[str, int]] = records
        counts = [sum(1 for _, l in self.records if l == i) for i in range(3)]
        logger.info(
            "GECVDataset: %d clips (positive=%d, neutral=%d, negative=%d) from %s",
            len(self.records), *counts, self.video_root,
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        clip_id, label = self.records[idx]
        try:
            sample = self._load_clip(clip_id)
        except Exception as e:
            if self.strict:
                raise
            logger.warning("Falling back to zeros for %s: %s", clip_id, e)
            sample = self._zero_fallback()
        sample["label"] = label
        sample["clip_id"] = clip_id
        return sample

    @property
    def all_clip_ids(self) -> List[str]:
        return [r[0] for r in self.records]

    @property
    def all_labels(self) -> List[int]:
        return [r[1] for r in self.records]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency CE weights. GECV split: 201/100/107."""
        labels = np.asarray(self.all_labels)
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        weights = labels.size / (len(CLASS_NAMES) * counts.clip(min=1))
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------

    def _load_clip(self, clip_id: str) -> Dict[str, torch.Tensor]:
        visual_path = self.cache_root / "visual" / f"{clip_id}.h5"
        audio_path = self.cache_root / "audio" / f"{clip_id}.h5"
        scene_path = self.cache_root / "scene" / f"{clip_id}.h5"

        with h5py.File(visual_path, "r") as f:
            features = f["features"][()]                                  # [T, N_max, 2048] fp16
            mask = f["person_mask"][()].astype(bool)                      # [T, N_max]

        with h5py.File(audio_path, "r") as f:
            audio = f["features"][()]                                     # [1024]
        with h5py.File(scene_path, "r") as f:
            scene = f["features"][()]                                     # [512]
        
        temporal = None
        if self.use_temporal:
            temporal_path = self.cache_root / "temporal" / f"{clip_id}.h5"
            with h5py.File(temporal_path, "r") as f:
                temporal = f["features"][()]                               # [768]
            if temporal.shape[-1] != self.temporal_dim:
                raise ValueError(
                    f"{clip_id}: temporal dim {temporal.shape[-1]} != "
                    f"expected {self.temporal_dim}"
                )

        # Sanity.
        if features.shape[-1] != self.per_person_dim:
            raise ValueError(
                f"{clip_id}: per_person_dim {features.shape[-1]} != "
                f"expected {self.per_person_dim}"
            )
        if audio.shape[-1] != self.audio_dim:
            raise ValueError(f"{clip_id}: audio dim {audio.shape[-1]} != {self.audio_dim}")
        if scene.shape[-1] != self.scene_dim:
            raise ValueError(f"{clip_id}: scene dim {scene.shape[-1]} != {self.scene_dim}")

        # Trim canonical N_max → slots actually used (any True in any frame).
        if self.trim_to_active_slots:
            active = mask.any(axis=0)                                     # [N_max]
            if not active.any():
                T = features.shape[0]
                features = np.zeros((T, 1, self.per_person_dim), dtype=np.float16)
                mask = np.zeros((T, 1), dtype=bool)
            else:
                features = features[:, active, :]
                mask = mask[:, active]

        out = {
            "per_person_features": torch.from_numpy(features.astype(np.float32)),
            "person_mask": torch.from_numpy(mask),
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "scene": torch.from_numpy(scene.astype(np.float32)),
        }
        if temporal is not None:                                              
            out["temporal"] = torch.from_numpy(temporal.astype(np.float32))   
        return out
    

    def _zero_fallback(self, T: int = 15) -> Dict[str, torch.Tensor]:
        out = {
            "per_person_features": torch.zeros(T, 1, self.per_person_dim),
            "person_mask": torch.zeros(T, 1, dtype=torch.bool),
            "audio": torch.zeros(self.audio_dim),
            "scene": torch.zeros(self.scene_dim),
        }
        if self.use_temporal:                                                 
            out["temporal"] = torch.zeros(self.temporal_dim)                  
        return out


# ----------------------------------------------------------------------

def gecv_collate(
    samples: List[Dict[str, Union[torch.Tensor, int, str]]],
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Pad per-person tensors to the batch's max N, stack everything."""
    B = len(samples)
    T = samples[0]["per_person_features"].shape[0]
    D = samples[0]["per_person_features"].shape[-1]
    N_max = max(s["per_person_features"].shape[1] for s in samples)

    per_person = torch.zeros(B, T, N_max, D)
    person_mask = torch.zeros(B, T, N_max, dtype=torch.bool)
    audio = torch.stack([s["audio"] for s in samples])
    scene = torch.stack([s["scene"] for s in samples])
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    clip_ids = [s["clip_id"] for s in samples]

    for i, s in enumerate(samples):
        n = s["per_person_features"].shape[1]
        if n > 0:
            per_person[i, :, :n] = s["per_person_features"]
            person_mask[i, :, :n] = s["person_mask"]

    out = {
        "per_person_features": per_person,
        "person_mask": person_mask,
        "audio": audio,
        "scene": scene,
        "label": labels,
        "clip_ids": clip_ids,
    }
    if "temporal" in samples[0]:                                          # ← new
        out["temporal"] = torch.stack([s["temporal"] for s in samples])   # ← new
    return out


# ----------------------------------------------------------------------

def stratified_kfold_splits(
    clip_ids: List[str],
    labels: List[int],
    k: int = 10,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Stratified k-fold split. Returns [(train_ids, val_ids), ...] of length k."""
    assert len(clip_ids) == len(labels)
    rng = np.random.default_rng(seed)

    by_label: Dict[int, List[str]] = {}
    for cid, lbl in zip(clip_ids, labels):
        by_label.setdefault(lbl, []).append(cid)
    for lbl in by_label:
        order = rng.permutation(len(by_label[lbl]))
        by_label[lbl] = [by_label[lbl][i] for i in order]

    folds: List[List[str]] = [[] for _ in range(k)]
    for lbl, cids in by_label.items():
        for i, cid in enumerate(cids):
            folds[i % k].append(cid)

    splits: List[Tuple[List[str], List[str]]] = []
    for f in range(k):
        val_ids = folds[f]
        train_ids = [c for fi, fc in enumerate(folds) if fi != f for c in fc]
        splits.append((train_ids, val_ids))
    return splits


# ----------------------------------------------------------------------

if __name__ == "__main__":
    """Metadata-only smoke test (does not require HDF5 caches)."""
    import os
    logging.basicConfig(level=logging.INFO)

    video_root = os.environ.get(
        "GECV_VIDEO_ROOT", "/path/to/datasets/GECV/GECV"
    )
    cache_root = os.environ.get(
        "GECV_CACHE_ROOT",
        "/path/to/datasets/gerhyperg_features/gecv",
    )
    ds = GECVDataset(video_root=video_root, cache_root=cache_root, strict=True)
    print(f"Total clips: {len(ds)}")
    print(f"First clip:  {ds.records[0]}")
    # print(f"Class weights (CE): {ds.class_weights().tolist()}")
    weights = ds.class_weights().tolist()
    print("Class weights (CE):")
    for name, w in zip(CLASS_NAMES, weights):
        print(f"  {name:8s}: {w:.4f}")

    splits = stratified_kfold_splits(ds.all_clip_ids, ds.all_labels, k=10, seed=42)
    print("\nStratified 10-fold:")
    for i, (tr, va) in enumerate(splits):
        val_labels = [CLASS_TO_IDX[next(c for c in CLASS_NAMES if cid.startswith(c + "_"))]
                      for cid in va]
        c = [val_labels.count(j) for j in range(3)]
        print(f"  fold {i}: train={len(tr):3d}  val={len(va):2d}  "
              f"val[pos/neu/neg]={c[0]}/{c[1]}/{c[2]}")