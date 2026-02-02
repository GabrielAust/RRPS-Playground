import numpy as np
import pytest

from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def test_action_mask_blocks_illegal() -> None:
    config = RRPSConfig(inventories_p0=(0, 1, 1), inventories_p1=(1, 1, 1))
    env = RRPSEnv(config)
    env.reset(seed=1)
    masks = env.action_masks()
    assert masks["p0"][0] == 0
    with pytest.raises(ValueError):
        env.step(0, 1)


def test_action_mask_random_resolves() -> None:
    config = RRPSConfig(
        inventories_p0=(0, 1, 1),
        inventories_p1=(1, 1, 1),
        illegal_action_mode="auto_mask_random",
    )
    env = RRPSEnv(config)
    env.reset(seed=1)
    obs, rewards, terminated, info = env.step(0, 1)
    assert "events_tail" in info
    assert obs["p0"].shape[0] == env.obs_dim()
