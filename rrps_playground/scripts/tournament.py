"""Run a round robin tournament across baseline agents."""
from __future__ import annotations

from rrps.agents import GreedyCounterLastAgent, RandomMaskedAgent, WSLSAgent
from rrps.cli import load_config_from_args
from rrps.env import RRPSEnv
from rrps.league import League
from rrps.match import BestOf
from rrps.persona import PersonaAgent, PersonaConfig


def main() -> None:
    config = load_config_from_args(description=__doc__)
    env = RRPSEnv(config)
    league = League(env)
    league.add_agent("random", RandomMaskedAgent())
    league.add_agent("greedy", GreedyCounterLastAgent())
    league.add_agent("wsls", WSLSAgent())

    personas = {
        "stoic": PersonaConfig(temperature=0.4, entropy=0.0, wsls_strength=0.2),
        "volatile": PersonaConfig(temperature=1.8, entropy=0.4, wsls_strength=0.3),
        "tilted": PersonaConfig(temperature=0.9, tilt_strength=0.5, wsls_strength=0.5),
        "counter": PersonaConfig(temperature=0.8, recency_alpha=0.7, wsls_strength=0.2),
        "forgetful": PersonaConfig(temperature=1.1, recency_alpha=0.1, entropy=0.2),
        "disciplined": PersonaConfig(temperature=0.6, anti_repeat_penalty=0.5),
        "stubborn": PersonaConfig(temperature=0.7, anti_repeat_penalty=-0.2),
        "paper_hoarder": PersonaConfig(
            temperature=0.9, conservation_bias=(0.0, 0.8, 0.0)
        ),
        "rock_hoarder": PersonaConfig(
            temperature=0.9, conservation_bias=(0.8, 0.0, 0.0)
        ),
        "scissor_hoarder": PersonaConfig(
            temperature=0.9, conservation_bias=(0.0, 0.0, 0.8)
        ),
        "entropy_low": PersonaConfig(temperature=0.7, entropy=0.0),
        "entropy_high": PersonaConfig(temperature=1.2, entropy=0.6),
        "wsls_high": PersonaConfig(temperature=0.8, wsls_strength=1.2),
        "wsls_low": PersonaConfig(temperature=1.0, wsls_strength=0.1),
        "anti_repeat": PersonaConfig(temperature=1.0, anti_repeat_penalty=0.8),
        "repeat_favor": PersonaConfig(temperature=0.9, anti_repeat_penalty=-0.4),
        "robust": PersonaConfig(temperature=1.0, robustness_mix=0.5),
        "sharp": PersonaConfig(temperature=0.6, robustness_mix=0.1),
        "reckless": PersonaConfig(temperature=1.6, entropy=0.3, robustness_mix=0.2),
        "patient": PersonaConfig(temperature=0.5, recency_alpha=0.5),
        "reactive": PersonaConfig(temperature=1.1, recency_alpha=0.8),
        "swingy": PersonaConfig(
            temperature=1.0, tilt_strength=0.7, anti_repeat_penalty=0.2
        ),
        "risk_averse": PersonaConfig(temperature=0.4, robustness_mix=0.7),
    }

    for name, persona in personas.items():
        league.add_agent(name, PersonaAgent(persona))

    leaderboard = league.round_robin(seed=123, match_format=BestOf(hands=5))
    for name, stats in leaderboard.items():
        print(f"{name}: {stats}")


if __name__ == "__main__":
    main()
