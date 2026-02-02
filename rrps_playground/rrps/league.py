"""League utilities for RRPS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .agents import Agent
from .env import RRPSEnv
from .match import BestOf, MatchResult, ScoreTo, play_match, seed_order


@dataclass
class League:
    """Opponent pool manager for RRPS."""

    env: RRPSEnv
    agents: Dict[str, Agent] = field(default_factory=dict)

    def add_agent(self, name: str, agent: Agent) -> None:
        """Add an agent to the pool."""
        self.agents[name] = agent

    def round_robin(
        self,
        seed: int | None = None,
        match_format: BestOf | ScoreTo | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run a round robin tournament and return leaderboard stats."""
        names = seed_order(self.agents.keys(), seed)
        stats = {name: {"wins": 0, "ties": 0, "losses": 0} for name in names}
        metrics = {name: {} for name in names}
        matches_played = {name: 0 for name in names}
        match_seed = seed
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i >= j:
                    continue
                match_seed = 0 if match_seed is None else match_seed + 1
                result = play_match(
                    self.env,
                    self.agents[name_i],
                    self.agents[name_j],
                    match_format=match_format,
                    seed=match_seed,
                )
                _update_match_stats(stats, name_i, name_j, result)
                _accumulate_metrics(metrics[name_i], result.metrics_p0)
                _accumulate_metrics(metrics[name_j], result.metrics_p1)
                matches_played[name_i] += 1
                matches_played[name_j] += 1
        leaderboard = {}
        for name, record in stats.items():
            total = record["wins"] + record["losses"] + record["ties"]
            total = max(total, 1)
            averages = {
                key: value / max(matches_played[name], 1)
                for key, value in metrics[name].items()
            }
            leaderboard[name] = {
                "win_rate": record["wins"] / total,
                "tie_rate": record["ties"] / total,
                "loss_rate": record["losses"] / total,
                **averages,
            }
        return leaderboard


def _update_match_stats(
    stats: Dict[str, Dict[str, int]],
    name_i: str,
    name_j: str,
    result: MatchResult,
) -> None:
    winner = result.winner()
    if winner == "p0":
        stats[name_i]["wins"] += 1
        stats[name_j]["losses"] += 1
    elif winner == "p1":
        stats[name_i]["losses"] += 1
        stats[name_j]["wins"] += 1
    else:
        stats[name_i]["ties"] += 1
        stats[name_j]["ties"] += 1


def _accumulate_metrics(metrics: Dict[str, float], update: Dict[str, float]) -> None:
    for key, value in update.items():
        metrics[key] = metrics.get(key, 0.0) + value
