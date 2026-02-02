"""Utility functions for RRPS."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def rps_outcome(action_p0: int, action_p1: int) -> int:
    """Return outcome for player 0: win=1, tie=0, loss=-1."""
    if action_p0 == action_p1:
        return 0
    if (action_p0 - action_p1) % 3 == 1:
        return 1
    return -1


def mask_from_counts(counts: Iterable[int]) -> np.ndarray:
    """Return a boolean mask for available actions."""
    counts_arr = np.asarray(list(counts), dtype=np.int32)
    return (counts_arr > 0).astype(np.int8)


def sample_masked(rng: np.random.Generator, mask: np.ndarray) -> int:
    """Sample an action from a 0/1 mask."""
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) == 0:
        raise ValueError("No valid actions available")
    return int(rng.choice(valid_indices))


def update_counts(counts: Tuple[int, int, int], action: int) -> Tuple[int, int, int]:
    """Return updated counts after using action."""
    counts_list = list(counts)
    counts_list[action] -= 1
    return tuple(counts_list)
