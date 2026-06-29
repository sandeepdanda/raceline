"""Seed every RNG that touches a run, so results are reproducible.

Seeds Python, NumPy, and Torch in one call; the Gymnasium env is seeded separately via
``reset(seed=...)`` since modern Gymnasium has no standalone ``env.seed()``.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
