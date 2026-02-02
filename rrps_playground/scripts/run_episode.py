"""Run a single episode with step-by-step logging."""
from __future__ import annotations

from rrps.agents import GreedyCounterLastAgent, RandomMaskedAgent
from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def main() -> None:
    config = RRPSConfig()
    env = RRPSEnv(config)
    agent_p0 = RandomMaskedAgent()
    agent_p1 = GreedyCounterLastAgent()

    obs = env.reset(seed=config.seed)
    step = 0
    while True:
        masks = env.action_masks()
        action_p0 = agent_p0.act(obs["p0"], masks["p0"], env.rng)
        action_p1 = agent_p1.act(obs["p1"], masks["p1"], env.rng)
        obs, rewards, terminated, info = env.step(action_p0, action_p1)
        print(
            f"Step {step}: a0={action_p0} a1={action_p1} rewards={rewards} masks={masks}"
        )
        print(f"Events tail: {info['events_tail']}")
        step += 1
        if terminated:
            break


if __name__ == "__main__":
    main()
