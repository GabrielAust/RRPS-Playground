import numpy as np

from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def test_env_reset_and_step() -> None:
    config = RRPSConfig(history_len=2)
    env = RRPSEnv(config)
    obs = env.reset(seed=123)
    assert set(obs.keys()) == {"p0", "p1"}
    assert obs["p0"].shape[0] == env.obs_dim()

    masks = env.action_masks()
    action_p0 = int(np.flatnonzero(masks["p0"])[0])
    action_p1 = int(np.flatnonzero(masks["p1"])[0])
    obs, rewards, terminated, info = env.step(action_p0, action_p1)
    assert isinstance(rewards[0], float)
    assert isinstance(terminated, bool)
    assert "events_tail" in info
