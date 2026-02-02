"""Collect self-play transitions into a simple buffer."""
from __future__ import annotations

from rrps.agents import RandomMaskedAgent
from rrps.buffer import Transition, TransitionBuffer
from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def main() -> None:
    env = RRPSEnv(RRPSConfig())
    agent = RandomMaskedAgent()
    buffer = TransitionBuffer()

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
            ),
            Transition(
                obs=obs["p1"],
                action=action_opp,
                reward=rewards[1],
                next_obs=next_obs["p1"],
                done=terminated,
                mask=masks["p1"],
            ),
        )
        obs = next_obs
        if terminated:
            break
    print(f"Collected {len(buffer.transitions)} transitions")


if __name__ == "__main__":
    main()
