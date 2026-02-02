"""Helpers for loading RRPS configs from CLI arguments."""
from __future__ import annotations

import argparse
from typing import Sequence

from .config import (
    RRPSConfig,
    apply_config_overrides,
    find_default_config,
    load_config_file,
)

ILLEGAL_ACTION_MODES = ("error", "auto_mask_random", "forfeit_round")


def load_config_from_args(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> RRPSConfig:
    """Parse CLI arguments and return a configured RRPSConfig."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        help="Path to config.json/config.yaml (defaults to config.{json,yaml,yml})",
    )
    parser.add_argument("--seed", type=int, help="RNG seed for the episode")
    parser.add_argument(
        "--counts",
        type=int,
        nargs=3,
        metavar=("ROCK", "PAPER", "SCISSORS"),
        help="Inventory counts for both players (three ints)",
    )
    parser.add_argument("--history-len", type=int, help="Observation history length")
    parser.add_argument(
        "--illegal-action-mode",
        choices=ILLEGAL_ACTION_MODES,
        help="Behavior when an illegal action is selected",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        default_path = find_default_config()
        config_path = str(default_path) if default_path is not None else None

    config = RRPSConfig()
    if config_path is not None:
        config = apply_config_overrides(config, load_config_file(config_path))

    overrides: dict[str, object] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.counts is not None:
        overrides["counts"] = args.counts
    if args.history_len is not None:
        overrides["history_len"] = args.history_len
    if args.illegal_action_mode is not None:
        overrides["illegal_action_mode"] = args.illegal_action_mode
    if overrides:
        config = apply_config_overrides(config, overrides)

    config.validate()
    return config
