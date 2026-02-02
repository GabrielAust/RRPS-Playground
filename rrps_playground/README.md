# RRPS Playground

RRPS (Rock-Restricted Paper-Scissors) is Rock-Paper-Scissors with finite inventories. Each player starts with a fixed number of Rock/Paper/Scissors tokens. On each round both players choose from their remaining inventory, reveal simultaneously, score (+1/0/-1), and remove the used symbol. The episode ends when the inventory is depleted or the max round limit is reached.

## Layout

```
rrps_playground/
  rrps/
    config.py
    env.py
    agents.py
    league.py
    eval.py
    utils.py
  scripts/
  tests/
```

## Quick start

Install dependencies:

```
pip install -e .
```

Run a single episode with logging:

```
python scripts/run_episode.py
```

Run a round-robin tournament across baseline agents:

```
python scripts/tournament.py
```

Collect self-play transitions:

```
python scripts/selfplay_collect.py
```

Run tests:

```
pytest
```

## Environment API

The environment is Gymnasium-like but has no external dependency. Use `reset()` and `step()` and query `action_masks()`.

```python
from rrps.config import RRPSConfig
from rrps.env import RRPSEnv

config = RRPSConfig(history_len=3)
env = RRPSEnv(config)
obs = env.reset(seed=123)
mask = env.action_masks()["p0"]
```

Observations are configurable but default to:

- Own remaining counts (3 floats)
- History of last `history_len` rounds (one-hot my action + opponent action per round)
- Normalized round index

## Creating your own agent

Agents implement the `act(obs, mask, rng)` interface:

```python
import numpy as np
from rrps.types import Agent

class AlwaysRockAgent(Agent):
    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        if mask[0] == 1:
            return 0
        return int(rng.choice(np.flatnonzero(mask)))
```

## Extensibility

Future toggles (signals, challenges, bets, commitments, noisy tells) are already in `RRPSConfig`. The environment records an event log per round with placeholders for those fields so you can extend game logic without changing the observation or evaluation APIs. When adding new mechanics:

1. Populate the placeholder fields in the event log.
2. Update the `info` dict to surface any new metadata.
3. Keep existing observation toggles stable for backward compatibility.

## Coding standards

- Python 3.10+
- `dataclasses` + typing throughout
- Numpy for numerics
- No heavyweight dependencies
