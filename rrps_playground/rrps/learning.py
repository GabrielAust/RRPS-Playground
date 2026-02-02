"""Learning utilities for RRPS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .config import RRPSConfig
from .types import Agent


State = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


def build_state(obs: np.ndarray, config: RRPSConfig) -> State:
    """Build a discrete state from observation counts."""
    obs = np.asarray(obs, dtype=np.float32)
    idx = 0
    if config.include_self_counts:
        counts_self = tuple(int(round(x)) for x in obs[idx : idx + 3])
        idx += 3
    else:
        counts_self = (-1, -1, -1)
    if config.include_opponent_counts:
        counts_opp = tuple(int(round(x)) for x in obs[idx : idx + 3])
    else:
        counts_opp = (-1, -1, -1)
    return counts_self, counts_opp


def _valid_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def _select_best_action(
    q_values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> int:
    valid = _valid_indices(mask)
    if valid.size == 0:
        raise ValueError("No valid actions available")
    masked_q = q_values[valid]
    best_value = np.max(masked_q)
    best_actions = valid[np.isclose(masked_q, best_value)]
    return int(rng.choice(best_actions))


@dataclass
class QTablePolicy(Agent):
    """Greedy policy derived from a Q-table."""

    q_table: Dict[State, np.ndarray]
    config: RRPSConfig

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        state = build_state(obs, self.config)
        q_values = self.q_table.get(state)
        if q_values is None:
            q_values = np.zeros(3, dtype=np.float64)
        return _select_best_action(q_values, mask, rng)


@dataclass
class TabularQLearner:
    """Tabular Q-learning with masked action selection."""

    config: RRPSConfig
    alpha: float = 0.4
    gamma: float = 0.95
    epsilon: float = 0.2
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.995
    q_table: Dict[State, np.ndarray] = field(default_factory=dict)

    def policy(self) -> QTablePolicy:
        return QTablePolicy(self.q_table, self.config)

    def _get_q(self, state: State) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3, dtype=np.float64)
        return self.q_table[state]

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            valid = _valid_indices(mask)
            if valid.size == 0:
                raise ValueError("No valid actions available")
            return int(rng.choice(valid))
        q_values = self._get_q(build_state(obs, self.config))
        return _select_best_action(q_values, mask, rng)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        state = build_state(obs, self.config)
        next_state = build_state(next_obs, self.config)
        q_values = self._get_q(state)
        next_q = self._get_q(next_state)
        valid_next = _valid_indices(next_mask)
        if done or valid_next.size == 0:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(next_q[valid_next]))
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_exploration(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
