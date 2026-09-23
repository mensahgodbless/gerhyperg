"""
Reproducibility utilities.
Seeds every source of randomness 
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        # Shouldn't happen — worker_init_fn only called in workers.
        return

    base_seed = worker_info.seed % (2 ** 32)
    # Per-worker offsets so different workers don't shadow each other.
    random.seed(base_seed + worker_id)
    np.random.seed((base_seed + worker_id) % (2 ** 32))


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g