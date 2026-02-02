"""Persona-based agents with adjustable behavioral knobs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .types import Agent
from .utils import rps_outcome, sample_masked


@dataclass(frozen=True)
class PersonaConfig:
    """Configuration knobs for PersonaAgent behavior."""

    temperature: float = 1.0
    entropy: float = 0.0
    wsls_strength: float = 0.6
    anti_repeat_penalty: float = 0.0
    tilt_strength: float = 0.0
    recency_alpha: float = 0.4
    conservation_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robustness_mix: float = 0.0


@dataclass
class PersonaAgent(Agent):
    """Agent that blends heuristics into action scores then samples."""

    config: PersonaConfig

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        scores, loss_streak = self._score_actions(obs, rng)
        if scores.shape != mask.shape:
            raise ValueError("Score vector shape must match mask shape")
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            raise ValueError("No valid actions available")
        temperature = self._tilted_temperature(loss_streak)
        if temperature <= 0:
            best_index = valid_indices[int(np.argmax(scores[valid_indices]))]
            return int(best_index)
        valid_scores = scores[valid_indices] / temperature
        valid_scores -= np.max(valid_scores)
        exp_scores = np.exp(valid_scores)
        probs = exp_scores / exp_scores.sum()
        return int(rng.choice(valid_indices, p=probs))

    def _tilted_temperature(self, loss_streak: int) -> float:
        if loss_streak <= 0:
            return self.config.temperature
        return self.config.temperature * (1.0 + self.config.tilt_strength * loss_streak)

    def _score_actions(
        self, obs: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, int]:
        counts_self, history = _parse_observation(obs)
        opponent_dist = _opponent_distribution(history, self.config.recency_alpha)
        scores = _expected_payoffs(opponent_dist)
        last_action, last_outcome, loss_streak = _last_action_outcome(history)
        if last_action is not None and last_outcome is not None:
            scores = _apply_wsls(scores, last_action, last_outcome, self.config)
            scores[last_action] -= self.config.anti_repeat_penalty
        scores = _apply_conservation(scores, counts_self, history, self.config)
        scores *= 1.0 - self.config.robustness_mix
        if self.config.entropy > 0:
            scores = scores + rng.normal(0.0, self.config.entropy, size=3)
        return scores, loss_streak


def _parse_observation(
    obs: np.ndarray,
) -> tuple[Optional[np.ndarray], list[tuple[Optional[int], Optional[int]]]]:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.size < 1:
        return None, []
    counts_len = (obs.size - 1) % 6
    if counts_len not in {0, 3, 6}:
        counts_len = 0
    base = obs[:-1]
    counts_self: Optional[np.ndarray] = None
    if counts_len >= 3:
        counts_self = base[:3]
    history_section = base[counts_len:]
    history: list[tuple[Optional[int], Optional[int]]] = []
    if history_section.size == 0:
        return counts_self, history
    history_len = history_section.size // 6
    if history_len == 0:
        return counts_self, history
    history_matrix = history_section[: history_len * 6].reshape(history_len, 6)
    for row in history_matrix:
        self_one_hot = row[:3]
        opp_one_hot = row[3:]
        if self_one_hot.sum() == 0 and opp_one_hot.sum() == 0:
            history.append((None, None))
            continue
        self_action = (
            int(np.argmax(self_one_hot)) if self_one_hot.sum() > 0 else None
        )
        opp_action = int(np.argmax(opp_one_hot)) if opp_one_hot.sum() > 0 else None
        history.append((self_action, opp_action))
    return counts_self, history


def _opponent_distribution(
    history: Iterable[tuple[Optional[int], Optional[int]]],
    alpha: float,
) -> np.ndarray:
    dist = np.full(3, 1.0 / 3.0)
    if alpha <= 0:
        return dist
    for _self_action, opp_action in history:
        if opp_action is None:
            continue
        one_hot = np.zeros(3)
        one_hot[opp_action] = 1.0
        dist = (1.0 - alpha) * dist + alpha * one_hot
    return dist


def _expected_payoffs(opponent_dist: np.ndarray) -> np.ndarray:
    scores = np.zeros(3)
    for action in range(3):
        payoff = 0.0
        for opp_action in range(3):
            payoff += opponent_dist[opp_action] * rps_outcome(action, opp_action)
        scores[action] = payoff
    return scores


def _last_action_outcome(
    history: list[tuple[Optional[int], Optional[int]]]
) -> tuple[Optional[int], Optional[int], int]:
    loss_streak = 0
    last_action = None
    last_outcome = None
    for self_action, opp_action in reversed(history):
        if self_action is None or opp_action is None:
            continue
        last_action = self_action
        last_outcome = rps_outcome(self_action, opp_action)
        break
    if last_outcome is None:
        return last_action, last_outcome, loss_streak
    for self_action, opp_action in reversed(history):
        if self_action is None or opp_action is None:
            continue
        outcome = rps_outcome(self_action, opp_action)
        if outcome == -1:
            loss_streak += 1
        else:
            break
    return last_action, last_outcome, loss_streak


def _apply_wsls(
    scores: np.ndarray,
    last_action: int,
    outcome: int,
    config: PersonaConfig,
) -> np.ndarray:
    scores = np.array(scores, dtype=np.float64)
    strength = config.wsls_strength
    if strength == 0:
        return scores
    if outcome == 1:
        scores[last_action] += strength
    elif outcome == -1:
        scores[last_action] -= strength
        for action in range(3):
            if action != last_action:
                scores[action] += strength / 2.0
    return scores


def _apply_conservation(
    scores: np.ndarray,
    counts_self: Optional[np.ndarray],
    history: list[tuple[Optional[int], Optional[int]]],
    config: PersonaConfig,
) -> np.ndarray:
    scores = np.array(scores, dtype=np.float64)
    bias = np.asarray(config.conservation_bias, dtype=np.float64)
    if bias.shape != (3,):
        raise ValueError("conservation_bias must be length 3")
    if np.allclose(bias, 0.0):
        return scores
    remaining_fraction = np.ones(3)
    if counts_self is not None:
        used = np.zeros(3)
        for action, _opp_action in history:
            if action is not None:
                used[action] += 1
        initial = counts_self + used
        with np.errstate(divide="ignore", invalid="ignore"):
            remaining_fraction = np.where(initial > 0, counts_self / initial, 0.0)
    penalty = bias * (1.0 - remaining_fraction)
    return scores - penalty
