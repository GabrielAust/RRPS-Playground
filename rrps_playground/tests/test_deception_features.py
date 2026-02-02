import numpy as np

from rrps.config import RRPSConfig
from rrps.env import RRPSEnv


def test_signals_phase_and_logging() -> None:
    config = RRPSConfig(enable_signals=True)
    env = RRPSEnv(config)
    env.reset(seed=1)
    assert env.phase == "signal"

    env.set_signals(2, 1)
    assert env.phase == "play"
    masks = env.action_masks()
    action_p0 = int(np.flatnonzero(masks["p0"])[0])
    action_p1 = int(np.flatnonzero(masks["p1"])[1])
    _, _, _, info = env.step(action_p0, action_p1)
    event = info["events_tail"][0]
    assert event["signal_p0"] == 2
    assert event["signal_p1"] == 1
    assert info["phase"] == "signal"


def test_challenge_rewards_correct_and_incorrect() -> None:
    config = RRPSConfig(
        enable_signals=True,
        enable_challenges=True,
        challenge_cost=0.25,
        challenge_penalty=0.5,
    )
    env = RRPSEnv(config)
    env.reset(seed=2)

    _, rewards, _, _ = env.step(
        1,
        2,
        signal_p0=0,
        signal_p1=2,
        challenge_p1=True,
    )
    assert rewards == (-1.5, 0.75)

    _, rewards, _, _ = env.step(
        0,
        2,
        signal_p0=0,
        signal_p1=2,
        challenge_p1=True,
    )
    assert rewards == (1.0, -1.75)


def test_side_bets_commitments_and_tells_logging() -> None:
    config = RRPSConfig(
        enable_side_bets=True,
        enable_commitments=True,
        enable_noisy_tells=True,
    )
    env = RRPSEnv(config)
    env.reset(seed=3)
    _, _, _, info = env.step(
        0,
        1,
        bet_p0={"bet": "up"},
        bet_p1={"bet": "down"},
        commitment_p0="rock",
        commitment_p1="paper",
        tell_p0=0.2,
        tell_p1=0.8,
    )
    event = info["events_tail"][0]
    assert event["bet_p0"] == {"bet": "up"}
    assert event["bet_p1"] == {"bet": "down"}
    assert event["commitment_p0"] == "rock"
    assert event["commitment_p1"] == "paper"
    assert event["tell_p0"] == 0.2
    assert event["tell_p1"] == 0.8
