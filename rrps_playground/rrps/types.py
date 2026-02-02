"""Shared types for RRPS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

import numpy as np

ObsDict = Dict[str, np.ndarray]
RewardTuple = Tuple[float, float]


@dataclass(frozen=True)
class StepResult:
    """Container for step outputs."""

    obs: ObsDict
    rewards: RewardTuple
    terminated: bool
    info: Dict[str, object]


class Agent(Protocol):
    """Protocol for agents."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        """Select an action given observation, mask, and RNG."""
        ...
