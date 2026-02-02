"""League utilities for RRPS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .agents import Agent
from .env import RRPSEnv
from .eval import play_episode


@dataclass
class League:
    """Opponent pool manager for RRPS."""

    env: RRPSEnv
    agents: Dict[str, Agent] = field(default_factory=dict)

    def add_agent(self, name: str, agent: Agent) -> None:
        """Add an agent to the pool."""
        self.agents[name] = agent

    def round_robin(self, seed: int | None = None) -> Dict[str, Dict[str, float]]:
        """Run a round robin tournament and return leaderboard stats."""
        names = list(self.agents.keys())
        stats = {name: {"wins": 0, "ties": 0, "losses": 0} for name in names}
        match_seed = seed
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i >= j:
                    continue
                match_seed = 0 if match_seed is None else match_seed + 1
                result = play_episode(
                    self.env, self.agents[name_i], self.agents[name_j], seed=match_seed
                )
                if result.total_reward_p0 > result.total_reward_p1:
                    stats[name_i]["wins"] += 1
                    stats[name_j]["losses"] += 1
                elif result.total_reward_p0 < result.total_reward_p1:
                    stats[name_i]["losses"] += 1
                    stats[name_j]["wins"] += 1
                else:
                    stats[name_i]["ties"] += 1
                    stats[name_j]["ties"] += 1
        leaderboard = {}
        for name, record in stats.items():
            total = record["wins"] + record["losses"] + record["ties"]
            total = max(total, 1)
            leaderboard[name] = {
                "win_rate": record["wins"] / total,
                "tie_rate": record["ties"] / total,
                "loss_rate": record["losses"] / total,
            }
        return leaderboard
