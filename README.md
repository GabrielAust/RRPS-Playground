# RRPS-Playground

RRPS-Playground is a small Python playground for **Rock-Restricted Paper-Scissors (RRPS)**, a Rock–Paper–Scissors variant where each player has a finite inventory of moves. The repository contains a lightweight environment, baseline agents, evaluation helpers, scripts, and tests.

## Quick start

```bash
cd rrps_playground
pip install -e .
```

Run a sample episode:

```bash
python scripts/run_episode.py
```

Run a round-robin tournament:

```bash
python scripts/tournament.py
```

Collect self-play transitions:

```bash
python scripts/selfplay_collect.py
```

Run tests:

```bash
pytest
```

## Repository contents (every file)

The table below documents every file in this repository (excluding `.git`).

| Path | Description |
| --- | --- |
| `README.md` | Top-level overview, setup, and file catalog (this file). |
| `rrps_playground/README.md` | Package-level README with environment API details, agent example, and extension notes. |
| `rrps_playground/pyproject.toml` | Project metadata, dependencies, and pytest configuration for the RRPS package. |
| `rrps_playground/rrps/__init__.py` | Package exports for configs, environment, and baseline agents. |
| `rrps_playground/rrps/agents.py` | Baseline agent implementations (random masked, greedy counter, WSLS) and helper to read last actions from observations. |
| `rrps_playground/rrps/config.py` | `RRPSConfig` dataclass defining inventories, observation toggles, reward settings, and future feature flags. |
| `rrps_playground/rrps/env.py` | Core environment implementation: action masking, step/reset logic, observations, rewards, and event logging. |
| `rrps_playground/rrps/eval.py` | Episode and multi-episode evaluation helpers to compute win/tie/loss rates. |
| `rrps_playground/rrps/league.py` | League helper to manage agents and run round-robin tournaments. |
| `rrps_playground/rrps/types.py` | Shared type aliases, `StepResult` container, and the `Agent` protocol. |
| `rrps_playground/rrps/utils.py` | Utility functions for outcomes, masks, sampling, and inventory updates. |
| `rrps_playground/scripts/run_episode.py` | Script that runs a single episode with verbose step logging. |
| `rrps_playground/scripts/selfplay_collect.py` | Script that collects self-play transitions into an in-memory buffer. |
| `rrps_playground/scripts/tournament.py` | Script that runs a round-robin tournament across baseline agents. |
| `rrps_playground/tests/test_action_mask.py` | Tests for action masking and illegal action handling modes. |
| `rrps_playground/tests/test_determinism.py` | Test verifying deterministic outcomes with identical seeds. |
| `rrps_playground/tests/test_env_basic.py` | Basic environment reset/step smoke test for observation shape and info fields. |

## Notes

- The core API is in `rrps_playground/rrps/env.py`, with configuration in `rrps_playground/rrps/config.py`.
- For deeper usage documentation and extensibility notes, see `rrps_playground/README.md`.
