"""Collect self-play transitions into a simple buffer."""
from __future__ import annotations

from rrps.agents import RandomMaskedAgent
from rrps.buffer import Transition, TransitionBuffer
from rrps.cli import load_config_from_args
from rrps.env import RRPSEnv


def main() -> None:
    config = load_config_from_args(description=__doc__)
    env = RRPSEnv(config)
    agent = RandomMaskedAgent()
    buffer = TransitionBuffer()

    obs = env.reset(seed=env.config.seed)
    while True:
        masks = env.action_masks()
        action = agent.act(obs["p0"], masks["p0"], env.rng)
        action_opp = agent.act(obs["p1"], masks["p1"], env.rng)
        next_obs, rewards, terminated, _info = env.step(action, action_opp)
        event = _info.get("events_tail", [{}])[0]
        buffer.add(
            Transition(
                obs=obs["p0"],
                mask=masks["p0"],
                action=action,
                reward=rewards[0],
                next_obs=next_obs["p0"],
                done=terminated,
                info_small=_info_small(event, "p0", _info),
            ),
            Transition(
                obs=obs["p1"],
                mask=masks["p1"],
                action=action_opp,
                reward=rewards[1],
                next_obs=next_obs["p1"],
                done=terminated,
                info_small=_info_small(event, "p1", _info),
            ),
        )
        obs = next_obs
        if terminated:
            break
    print(f"Collected {len(buffer.transitions)} transitions")


def _info_small(
    event: dict[str, object], player: str, info: dict[str, object]
) -> dict[str, object]:
    outcome_p0 = event.get("outcome_p0")
    outcome = None
    if isinstance(outcome_p0, int):
        outcome = outcome_p0 if player == "p0" else -outcome_p0
    counts_self = event.get("counts_p0" if player == "p0" else "counts_p1")
    counts_opp = event.get("counts_p1" if player == "p0" else "counts_p0")
    return {
        "round": info.get("round"),
        "phase": info.get("phase"),
        "outcome": outcome,
        "counts_self": counts_self,
        "counts_opp": counts_opp,
    }


if __name__ == "__main__":
    main()
