"""Evaluation utilities for RRPS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .agents import Agent
from .env import RRPSEnv


@dataclass
class EpisodeResult:
    """Summary for a single episode."""

    total_reward_p0: float
    total_reward_p1: float
    rounds: int


def play_episode(
    env: RRPSEnv, agent_p0: Agent, agent_p1: Agent, seed: int | None = None
) -> EpisodeResult:
    """Play a single episode and return totals."""
    obs = env.reset(seed=seed)
    total_p0 = 0.0
    total_p1 = 0.0
    while True:
        masks = env.action_masks()
        action_p0 = agent_p0.act(obs["p0"], masks["p0"], env.rng)
        action_p1 = agent_p1.act(obs["p1"], masks["p1"], env.rng)
        obs, rewards, terminated, _info = env.step(action_p0, action_p1)
        total_p0 += rewards[0]
        total_p1 += rewards[1]
        if terminated:
            break
    return EpisodeResult(total_p0, total_p1, env.round_index)


def evaluate_win_rates(
    env: RRPSEnv,
    agent_p0: Agent,
    agent_p1: Agent,
    episodes: int,
    seed: int | None = None,
) -> Dict[str, float]:
    """Evaluate win/tie/loss rates over multiple episodes."""
    wins = ties = losses = 0
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        episode_seed = int(rng.integers(0, 2**32 - 1))
        result = play_episode(env, agent_p0, agent_p1, seed=episode_seed)
        if result.total_reward_p0 > result.total_reward_p1:
            wins += 1
        elif result.total_reward_p0 < result.total_reward_p1:
            losses += 1
        else:
            ties += 1
    total = float(episodes)
    return {
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "loss_rate": losses / total,
    }
