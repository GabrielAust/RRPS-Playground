"""Train a tabular Q-learning agent with self-play scaffolding."""
from __future__ import annotations

import argparse
from collections import deque
from typing import Iterable, List

import numpy as np

from rrps.agents import GreedyCounterLastAgent, RandomMaskedAgent, WSLSAgent
from rrps.buffer import Transition, TransitionBuffer
from rrps.cli import load_config_from_args
from rrps.config import apply_config_overrides
from rrps.env import RRPSEnv
from rrps.league import OpponentSnapshot, SelfPlayLeague
from rrps.learning import QTablePolicy, TabularQLearner
from rrps.persona import PersonaAgent, PersonaConfig


def _parse_training_args(
    argv: List[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--save-jsonl", type=str, default=None)
    parser.add_argument("--save-npz", type=str, default=None)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def _persona_suite() -> List[OpponentSnapshot]:
    personas: Iterable[OpponentSnapshot] = [
        OpponentSnapshot("random_masked", RandomMaskedAgent()),
        OpponentSnapshot("greedy_counter", GreedyCounterLastAgent()),
        OpponentSnapshot("wsls", WSLSAgent()),
        OpponentSnapshot("persona_default", PersonaAgent(PersonaConfig())),
        OpponentSnapshot(
            "persona_tilted",
            PersonaAgent(
                PersonaConfig(tilt_strength=0.8, wsls_strength=0.8, entropy=0.1)
            ),
        ),
        OpponentSnapshot(
            "persona_entropy",
            PersonaAgent(PersonaConfig(entropy=0.4, anti_repeat_penalty=0.2)),
        ),
    ]
    return list(personas)


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


def _maybe_snapshot(
    league: SelfPlayLeague,
    pool: deque[OpponentSnapshot],
    name: str,
    agent: QTablePolicy,
    pool_size: int,
) -> None:
    league.add_snapshot(name, agent, copy_agent=True)
    pool.append(league.opponents[-1])
    while len(pool) > pool_size:
        pool.popleft()
        league.opponents = list(pool)


def main(argv: List[str] | None = None) -> None:
    args, remaining = _parse_training_args(argv)
    config = load_config_from_args(remaining, description=__doc__)
    config = apply_config_overrides(
        config,
        {"include_opponent_counts": True, "include_history": False},
    )
    env = RRPSEnv(config)
    rng = np.random.default_rng(config.seed)
    learner = TabularQLearner(config)
    league = SelfPlayLeague(env)
    opponent_pool: deque[OpponentSnapshot] = deque()

    base_opponent = OpponentSnapshot("random_masked", RandomMaskedAgent())
    league.add_snapshot(base_opponent.name, base_opponent.agent, copy_agent=True)
    opponent_pool.append(league.opponents[-1])
    _maybe_snapshot(league, opponent_pool, "snapshot_0", learner.policy(), args.pool_size)

    buffer = TransitionBuffer()
    personas = _persona_suite()

    for episode in range(1, args.episodes + 1):
        snapshot = league.sample_opponent(rng)
        opponent = snapshot.agent
        obs = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
        while True:
            masks = env.action_masks()
            action = learner.act(obs["p0"], masks["p0"], rng)
            opp_action = opponent.act(obs["p1"], masks["p1"], env.rng)
            next_obs, rewards, terminated, info = env.step(action, opp_action)
            next_masks = env.action_masks()
            learner.update(
                obs["p0"],
                action,
                rewards[0],
                next_obs["p0"],
                terminated,
                next_masks["p0"],
            )
            event = info.get("events_tail", [{}])[0]
            buffer.add(
                Transition(
                    obs=obs["p0"],
                    mask=masks["p0"],
                    action=action,
                    reward=rewards[0],
                    next_obs=next_obs["p0"],
                    done=terminated,
                    info_small=_info_small(event, "p0", info),
                ),
                Transition(
                    obs=obs["p1"],
                    mask=masks["p1"],
                    action=opp_action,
                    reward=rewards[1],
                    next_obs=next_obs["p1"],
                    done=terminated,
                    info_small=_info_small(event, "p1", info),
                ),
            )
            obs = next_obs
            if terminated:
                break
        learner.decay_exploration()

        if args.snapshot_every and episode % args.snapshot_every == 0:
            _maybe_snapshot(
                league,
                opponent_pool,
                f"snapshot_{episode}",
                learner.policy(),
                args.pool_size,
            )

        if args.eval_every and episode % args.eval_every == 0:
            results = league.evaluate_personas(
                learner.policy(), personas, episodes=args.eval_episodes, seed=episode
            )
            win_vs_random = results["random_masked"]["win_rate"]
            print(
                f"[episode {episode}] win_rate_vs_random={win_vs_random:.2%} "
                f"epsilon={learner.epsilon:.3f}"
            )

    final_results = league.evaluate_personas(
        learner.policy(), personas, episodes=args.eval_episodes, seed=999
    )
    print("Final evaluation:")
    for name, metrics in final_results.items():
        print(f"  {name}: {metrics}")

    if args.save_jsonl:
        buffer.save_jsonl(args.save_jsonl)
    if args.save_npz:
        buffer.save_npz(args.save_npz)


if __name__ == "__main__":
    main()
