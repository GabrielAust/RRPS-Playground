from rrps.agents import RandomMaskedAgent
from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def test_determinism_same_seed() -> None:
    config = RRPSConfig(seed=42)
    env_a = RRPSEnv(config)
    env_b = RRPSEnv(config)
    agent_a0 = RandomMaskedAgent()
    agent_a1 = RandomMaskedAgent()
    agent_b0 = RandomMaskedAgent()
    agent_b1 = RandomMaskedAgent()

    obs_a = env_a.reset(seed=42)
    obs_b = env_b.reset(seed=42)

    while True:
        masks_a = env_a.action_masks()
        masks_b = env_b.action_masks()
        action_a0 = agent_a0.act(obs_a["p0"], masks_a["p0"], env_a.rng)
        action_a1 = agent_a1.act(obs_a["p1"], masks_a["p1"], env_a.rng)
        action_b0 = agent_b0.act(obs_b["p0"], masks_b["p0"], env_b.rng)
        action_b1 = agent_b1.act(obs_b["p1"], masks_b["p1"], env_b.rng)
        obs_a, rewards_a, term_a, _ = env_a.step(action_a0, action_a1)
        obs_b, rewards_b, term_b, _ = env_b.step(action_b0, action_b1)
        assert rewards_a == rewards_b
        assert env_a.events[-1] == env_b.events[-1]
        if term_a or term_b:
            break
