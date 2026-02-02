import numpy as np

from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def _step_once(env: RRPSEnv) -> dict:
    masks = env.action_masks()
    action_p0 = int(np.flatnonzero(masks["p0"])[0])
    action_p1 = int(np.flatnonzero(masks["p1"])[0])
    _, _, _, info = env.step(action_p0, action_p1)
    return info


def test_info_contract_fields_present() -> None:
    env = RRPSEnv(RRPSConfig())
    env.reset(seed=7)
    info = _step_once(env)
    assert "action_mask" in info
    assert "round" in info
    assert "events_tail" in info
    assert info["round"] == 0


def test_event_contract_placeholders_present() -> None:
    env = RRPSEnv(RRPSConfig())
    env.reset(seed=11)
    info = _step_once(env)
    event = info["events_tail"][0]
    placeholders = [
        "signal_p0",
        "signal_p1",
        "challenge_p0",
        "challenge_p1",
        "bet_p0",
        "bet_p1",
        "commitment_p0",
        "commitment_p1",
        "tell_p0",
        "tell_p1",
    ]
    for key in placeholders:
        assert key in event
