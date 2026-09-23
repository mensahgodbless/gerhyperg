"""
Stratified train/dev split utilities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np


def stratified_train_dev_split(
    clip_ids: List[str],
    labels: List[int],
    dev_ratio: float = 0.2,
    seed: int = 7,
) -> Tuple[List[str], List[str]]:
    
    if len(clip_ids) != len(labels):
        raise ValueError("clip_ids and labels must be the same length")
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in (0, 1), got {dev_ratio}")

    rng = np.random.default_rng(seed)

    by_label: dict[int, List[str]] = defaultdict(list)
    for cid, lbl in zip(clip_ids, labels):
        by_label[lbl].append(cid)

    train_ids: List[str] = []
    dev_ids: List[str] = []
    for lbl in sorted(by_label.keys()):
        ids_for_label = by_label[lbl]
        order = rng.permutation(len(ids_for_label))
        shuffled = [ids_for_label[i] for i in order]
        n_dev = max(1, int(round(len(shuffled) * dev_ratio)))
        dev_ids.extend(shuffled[:n_dev])
        train_ids.extend(shuffled[n_dev:])

    assert set(train_ids).isdisjoint(set(dev_ids))
    assert set(train_ids) | set(dev_ids) == set(clip_ids)
    return train_ids, dev_ids
