"""Match formats for RRPS games."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .agents import Agent
from .env import RRPSEnv
from .eval import compute_action_metrics, play_episode


@dataclass(frozen=True)
class BestOf:
    """Best-of-N hands match format."""

    hands: int

    def wins_needed(self) -> int:
        return self.hands // 2 + 1


@dataclass(frozen=True)
class ScoreTo:
    """Score-to-K wins format with optional hand cap."""

    target_wins: int
    max_hands: Optional[int] = None


@dataclass
class MatchResult:
    """Summary for a match made of multiple hands."""

    wins_p0: int
    wins_p1: int
    ties: int
    total_reward_p0: float
    total_reward_p1: float
    hands: int
    metrics_p0: Dict[str, float]
    metrics_p1: Dict[str, float]

    def winner(self) -> str:
        if self.wins_p0 > self.wins_p1:
            return "p0"
        if self.wins_p1 > self.wins_p0:
            return "p1"
        if self.total_reward_p0 > self.total_reward_p1:
            return "p0"
        if self.total_reward_p1 > self.total_reward_p0:
            return "p1"
        return "tie"


def seed_order(names: Iterable[str], seed: int | None = None) -> List[str]:
    """Return a seeded ordering of names for tournament seeding."""
    rng = np.random.default_rng(seed)
    names_list = list(names)
    rng.shuffle(names_list)
    return names_list


def play_match(
    env: RRPSEnv,
    agent_p0: Agent,
    agent_p1: Agent,
    match_format: BestOf | ScoreTo | None = None,
    seed: int | None = None,
) -> MatchResult:
    """Play a match and return aggregated results and metrics."""
    rng = np.random.default_rng(seed)
    wins_p0 = wins_p1 = ties = 0
    total_p0 = total_p1 = 0.0
    metrics_p0: Dict[str, float] = {}
    metrics_p1: Dict[str, float] = {}
    hands_played = 0

    def update_metrics(metrics: Dict[str, float], update: Dict[str, float]) -> None:
        for key, value in update.items():
            metrics[key] = metrics.get(key, 0.0) + value

    def maybe_stop() -> bool:
        if isinstance(match_format, BestOf):
            wins_needed = match_format.wins_needed()
            return wins_p0 >= wins_needed or wins_p1 >= wins_needed
        if isinstance(match_format, ScoreTo):
            if wins_p0 >= match_format.target_wins or wins_p1 >= match_format.target_wins:
                return True
            if match_format.max_hands is not None:
                return hands_played >= match_format.max_hands
        return False

    while True:
        episode_seed = int(rng.integers(0, 2**32 - 1))
        result = play_episode(env, agent_p0, agent_p1, seed=episode_seed)
        metrics = compute_action_metrics(env.events, env.config)
        update_metrics(metrics_p0, metrics["p0"])
        update_metrics(metrics_p1, metrics["p1"])
        total_p0 += result.total_reward_p0
        total_p1 += result.total_reward_p1
        if result.total_reward_p0 > result.total_reward_p1:
            wins_p0 += 1
        elif result.total_reward_p0 < result.total_reward_p1:
            wins_p1 += 1
        else:
            ties += 1
        hands_played += 1
        if match_format is None:
            break
        if maybe_stop():
            break
        if isinstance(match_format, BestOf) and hands_played >= match_format.hands:
            break
    if hands_played > 0:
        metrics_p0 = {key: value / hands_played for key, value in metrics_p0.items()}
        metrics_p1 = {key: value / hands_played for key, value in metrics_p1.items()}
    return MatchResult(
        wins_p0=wins_p0,
        wins_p1=wins_p1,
        ties=ties,
        total_reward_p0=total_p0,
        total_reward_p1=total_p1,
        hands=hands_played,
        metrics_p0=metrics_p0,
        metrics_p1=metrics_p1,
    )
