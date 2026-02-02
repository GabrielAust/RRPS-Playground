"""Collect self-play transitions into a simple buffer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from rrps.agents import RandomMaskedAgent
from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    mask: np.ndarray


@dataclass
class Buffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)


def main() -> None:
    env = RRPSEnv(RRPSConfig())
    agent = RandomMaskedAgent()
    buffer = Buffer()

    obs = env.reset(seed=env.config.seed)
    while True:
        masks = env.action_masks()
        action = agent.act(obs["p0"], masks["p0"], env.rng)
        action_opp = agent.act(obs["p1"], masks["p1"], env.rng)
        next_obs, rewards, terminated, _info = env.step(action, action_opp)
        buffer.add(
            Transition(
                obs=obs["p0"],
                action=action,
                reward=rewards[0],
                next_obs=next_obs["p0"],
                done=terminated,
                mask=masks["p0"],
            )
        )
        obs = next_obs
        if terminated:
            break
    print(f"Collected {len(buffer.transitions)} transitions")


if __name__ == "__main__":
    main()
