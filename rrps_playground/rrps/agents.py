"""Baseline agents for RRPS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .types import Agent
from .utils import rps_outcome, sample_masked


def _last_actions_from_obs(obs: np.ndarray) -> tuple[Optional[int], Optional[int]]:
    """Extract last (self, opp) actions from obs history if available."""
    if obs.size < 7:
        return None, None
    history_slice = obs[-7:-1]
    if history_slice.size != 6:
        return None, None
    self_one_hot = history_slice[:3]
    opp_one_hot = history_slice[3:]
    if self_one_hot.sum() == 0 and opp_one_hot.sum() == 0:
        return None, None
    self_action = int(np.argmax(self_one_hot)) if self_one_hot.sum() > 0 else None
    opp_action = int(np.argmax(opp_one_hot)) if opp_one_hot.sum() > 0 else None
    return self_action, opp_action


@dataclass
class RandomMaskedAgent(Agent):
    """Random agent respecting action masks."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        return sample_masked(rng, mask)


@dataclass
class GreedyCounterLastAgent(Agent):
    """Agent that counters opponent's last action when possible."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        _, opp_action = _last_actions_from_obs(obs)
        if opp_action is None:
            return sample_masked(rng, mask)
        counter = (opp_action + 1) % 3
        if mask[counter] == 1:
            return int(counter)
        return sample_masked(rng, mask)


@dataclass
class WSLSAgent(Agent):
    """Win-stay / lose-shift agent based on last outcome."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        self_action, opp_action = _last_actions_from_obs(obs)
        if self_action is None or opp_action is None:
            return sample_masked(rng, mask)
        outcome = rps_outcome(self_action, opp_action)
        if outcome == 1:
            if mask[self_action] == 1:
                return int(self_action)
            return sample_masked(rng, mask)
        valid_indices = np.flatnonzero(mask)
        alternatives = [idx for idx in valid_indices if idx != self_action]
        if alternatives:
            return int(rng.choice(alternatives))
        return sample_masked(rng, mask)
