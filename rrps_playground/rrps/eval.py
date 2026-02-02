"""Evaluation utilities for RRPS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .agents import Agent
from .config import RRPSConfig
from .env import RRPSEnv


@dataclass
class EpisodeResult:
    """Summary for a single episode."""

    total_reward_p0: float
    total_reward_p1: float
    rounds: int


@dataclass
class ActionMetrics:
    """Behavioral metrics for a single player."""

    action_entropy: float
    repeat_rate: float
    wsls_rate: float
    reaction_to_loss: float
    endgame_efficiency: float
    wasted_tokens: float


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


def compute_action_metrics(
    events: list[Dict[str, object]],
    config: RRPSConfig,
) -> Dict[str, Dict[str, float]]:
    """Compute behavioral metrics for both players from event logs."""
    return {
        "p0": _metrics_for_player(events, "p0", config.inventories_p0),
        "p1": _metrics_for_player(events, "p1", config.inventories_p1),
    }


def _metrics_for_player(
    events: list[Dict[str, object]],
    player: str,
    initial_counts: Tuple[int, int, int],
) -> Dict[str, float]:
    actions, opp_actions, outcomes = _extract_actions(events, player)
    metrics = ActionMetrics(
        action_entropy=_action_entropy(actions),
        repeat_rate=_repeat_rate(actions),
        wsls_rate=_wsls_rate(actions, outcomes),
        reaction_to_loss=_reaction_to_loss(actions, opp_actions, outcomes),
        endgame_efficiency=_endgame_efficiency(events, player, initial_counts),
        wasted_tokens=_wasted_tokens(events, player),
    )
    return metrics.__dict__


def _extract_actions(
    events: list[Dict[str, object]],
    player: str,
) -> tuple[list[int], list[int], list[int]]:
    actions: list[int] = []
    opp_actions: list[int] = []
    outcomes: list[int] = []
    for event in events:
        action = event.get(f"action_{player}")
        opp_player = "p1" if player == "p0" else "p0"
        opp_action = event.get(f"action_{opp_player}")
        outcome_p0 = event.get("outcome_p0")
        if action is None or opp_action is None:
            continue
        if action == -1 or opp_action == -1:
            continue
        actions.append(int(action))
        opp_actions.append(int(opp_action))
        if outcome_p0 is None:
            outcomes.append(0)
        else:
            outcome_value = int(outcome_p0)
            outcomes.append(outcome_value if player == "p0" else -outcome_value)
    return actions, opp_actions, outcomes


def _action_entropy(actions: list[int]) -> float:
    if not actions:
        return 0.0
    counts = np.bincount(actions, minlength=3).astype(np.float64)
    probs = counts / counts.sum()
    entropy = 0.0
    for prob in probs:
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return float(entropy)


def _repeat_rate(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return repeats / (len(actions) - 1)


def _wsls_rate(actions: list[int], outcomes: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    wins_or_losses = 0
    successes = 0
    for i in range(1, len(actions)):
        outcome = outcomes[i - 1]
        if outcome == 0:
            continue
        wins_or_losses += 1
        if outcome == 1 and actions[i] == actions[i - 1]:
            successes += 1
        elif outcome == -1 and actions[i] != actions[i - 1]:
            successes += 1
    if wins_or_losses == 0:
        return 0.0
    return successes / wins_or_losses


def _reaction_to_loss(
    actions: list[int], opp_actions: list[int], outcomes: list[int]
) -> float:
    if len(actions) < 2:
        return 0.0
    opportunities = 0
    reacted = 0
    for i in range(1, len(actions)):
        if outcomes[i - 1] != -1:
            continue
        opportunities += 1
        counter = (opp_actions[i - 1] + 1) % 3
        if actions[i] == counter:
            reacted += 1
    if opportunities == 0:
        return 0.0
    return reacted / opportunities


def _endgame_efficiency(
    events: list[Dict[str, object]],
    player: str,
    initial_counts: Tuple[int, int, int],
) -> float:
    initial_total = sum(initial_counts)
    if initial_total == 0:
        return 0.0
    remaining = _wasted_tokens(events, player)
    return 1.0 - (remaining / initial_total)


def _wasted_tokens(events: list[Dict[str, object]], player: str) -> float:
    if not events:
        return 0.0
    last_event = events[-1]
    counts = last_event.get(f"counts_{player}")
    if counts is None:
        return 0.0
    return float(sum(counts))
