"""Run a round robin tournament across baseline agents."""
from __future__ import annotations

from rrps.agents import GreedyCounterLastAgent, RandomMaskedAgent, WSLSAgent
from rrps.config import RRPSConfig
from rrps.env import RRPSEnv
from rrps.league import League


def main() -> None:
    env = RRPSEnv(RRPSConfig())
    league = League(env)
    league.add_agent("random", RandomMaskedAgent())
    league.add_agent("greedy", GreedyCounterLastAgent())
    league.add_agent("wsls", WSLSAgent())

    leaderboard = league.round_robin(seed=123)
    for name, stats in leaderboard.items():
        print(f"{name}: {stats}")


if __name__ == "__main__":
    main()
