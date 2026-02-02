"""Baseline agents for RRPS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def extract_last_round(
    obs: np.ndarray,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (self_action, opp_action, outcome) from the most recent round."""
    self_action, opp_action = _last_actions_from_obs(obs)
    if self_action is None or opp_action is None:
        return self_action, opp_action, None
    return self_action, opp_action, rps_outcome(self_action, opp_action)


@dataclass
class MaskedSoftmaxAgent(Agent):
    """Agent base class that samples from masked softmax over action scores."""

    temperature: float = 1.0

    def score_actions(
        self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a score vector for actions."""
        raise NotImplementedError

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        scores = np.asarray(self.score_actions(obs, mask, rng), dtype=np.float64)
        if scores.shape != mask.shape:
            raise ValueError("Score vector shape must match mask shape")
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            raise ValueError("No valid actions available")
        if self.temperature <= 0:
            best_index = valid_indices[int(np.argmax(scores[valid_indices]))]
            return int(best_index)
        valid_scores = scores[valid_indices]
        scaled = valid_scores / self.temperature
        scaled -= np.max(scaled)
        exp_scores = np.exp(scaled)
        probs = exp_scores / exp_scores.sum()
        return int(rng.choice(valid_indices, p=probs))


@dataclass
class RandomMaskedAgent(Agent):
    """Random agent respecting action masks."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        return sample_masked(rng, mask)


@dataclass
class GreedyCounterLastAgent(Agent):
    """Agent that counters opponent's last action when possible."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        _, opp_action, _ = extract_last_round(obs)
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
        self_action, opp_action, outcome = extract_last_round(obs)
        if self_action is None or opp_action is None or outcome is None:
            return sample_masked(rng, mask)
        if outcome == 1:
            if mask[self_action] == 1:
                return int(self_action)
            return sample_masked(rng, mask)
        valid_indices = np.flatnonzero(mask)
        alternatives = [idx for idx in valid_indices if idx != self_action]
        if alternatives:
            return int(rng.choice(alternatives))
        return sample_masked(rng, mask)
