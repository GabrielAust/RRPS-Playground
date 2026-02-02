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

All scripts support optional CLI flags for configuration:

```
python scripts/run_episode.py --seed 7 --counts 3 3 3 --history-len 10 --illegal-action-mode auto_mask_random
```

You can also drop a `config.json`/`config.yaml` in the working directory or pass a path:

```
python scripts/tournament.py --config configs/experiment.json
```

Run a round-robin tournament across baseline agents:

```
python scripts/tournament.py
```

Collect self-play transitions:

```
python scripts/selfplay_collect.py
```

Train a tabular Q-learning baseline with league self-play:

```
python scripts/train_q_learning.py --episodes 2500 --eval-every 250
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

### Observation layout (exact indices)

Observations are flat `float32` vectors with a deterministic, index-stable layout. The layout is controlled by the
`include_*` flags and `history_len` in `RRPSConfig`. Indices below are **0-based** and apply to each player's
observation independently.

**Default layout (include_self_counts=True, include_opponent_counts=False, include_history=True, history_len=5)**

| Index range | Feature block | Details |
| --- | --- | --- |
| `0..2` | Self counts | Remaining `(rock, paper, scissors)` counts for the observing player. |
| `3..32` | History (5 rounds × 6) | For each of the last 5 rounds: `[self_one_hot(3), opp_one_hot(3)]`. Oldest first; padded with zeros if fewer than 5 rounds. |
| `33` | Round progress | `round_index / max_rounds` in `[0, 1]`. |

**General formula**

Let:

- `S = 3` if `include_self_counts` else `0`
- `O = 3` if `include_opponent_counts` else `0`
- `H = history_len * 6` if `include_history` else `0`
- `R = 1` (always present; normalized round index)

Then the observation length is `S + O + H + R` and the indices are assigned in this order:

1. **Self counts** (`S` values): indices `0 .. S-1`
2. **Opponent counts** (`O` values): indices `S .. S+O-1`
3. **History** (`H` values): indices `S+O .. S+O+H-1`
4. **Round progress** (`1` value): index `S+O+H`

**History block indexing**

When history is enabled, each round contributes 6 floats:

```
round i offset = S + O + i * 6
round i features:
  [offset + 0..2] -> self action one-hot (R, P, S)
  [offset + 3..5] -> opponent action one-hot (R, P, S)
```

Rounds are ordered from **oldest to newest** within the last `history_len` rounds. If there are fewer than
`history_len` rounds so far, leading rounds are padded with zeros (both players = all-zero one-hot).

Actions use the integer mapping **0=Rock, 1=Paper, 2=Scissors** throughout observations, masks, and logs.

### Action masks

`env.action_masks()` returns a dict with `{"p0": mask0, "p1": mask1}`. Each mask is a length-3 int8 vector aligned
to the `(Rock, Paper, Scissors)` action indices:

- `1` means the action is **legal/available** (remaining count > 0).
- `0` means the action is **illegal/unavailable** (remaining count == 0).

Masks are derived directly from each player's remaining inventory counts.

### Determinism guarantees

The environment is deterministic given:

- The same `RRPSConfig` (including inventories and `illegal_action_mode`),
- The same seed passed to `RRPSEnv(seed=...)` or `env.reset(seed=...)`,
- The same sequence of actions passed to `step()`.

When `illegal_action_mode="auto_mask_random"`, any illegal action is replaced by a sampled legal action using the
environment RNG (`numpy.random.default_rng`). This sampling is deterministic under the same seed and call order.

### Illegal action modes

The `illegal_action_mode` config controls how invalid actions are handled:

- `"error"`: Raise `ValueError` immediately when a player picks an unavailable or out-of-range action.
- `"auto_mask_random"`: Replace the illegal action with a random **legal** action sampled from the action mask.
  - If the mask has no legal actions (all zeros), the action resolves to `None` (no move).
- `"forfeit_round"`: Treat the illegal action as **no move** (`None`) for that player.

Resolution effects (per step):

- If both players resolve to `None`, the round is a tie and no inventories change.
- If one player resolves to `None` and the other has a legal action, the legal action player wins the round and
  only their inventory is decremented.
- If both actions are legal, both inventories decrement and outcome follows standard RPS rules.

### Event log schema

Each call to `step()` appends one event dict to `env.events` and includes the most recent entry in
`info["events_tail"]`. The schema is stable and includes placeholder keys for future mechanics:

```
{
  "round_index": int,                 # 0-based round index before increment
  "phase": str,                       # "play" (signals set the next phase to "signal" in info)
  "action_p0": int | None,            # resolved action (0/1/2) or None if forfeited/none
  "action_p1": int | None,
  "outcome_p0": int,                  # win=1, tie=0, loss=-1
  "counts_p0": (int, int, int),       # remaining inventory after the round
  "counts_p1": (int, int, int),
  "signal_p0": object | None,         # placeholder, default None
  "signal_p1": object | None,
  "challenge_p0": object | None,      # placeholder, default None
  "challenge_p1": object | None,
  "bet_p0": object | None,            # placeholder, default None
  "bet_p1": object | None,
  "commitment_p0": object | None,     # placeholder, default None
  "commitment_p1": object | None,
  "tell_p0": object | None,           # placeholder, default None
  "tell_p1": object | None
}
```

These fields exist to preserve log/observation compatibility; they are populated when the matching
`enable_*` flags are set and remain `None` otherwise.

### Signals and challenges (deception-lite)

When `enable_signals=True`, the environment alternates between a lightweight signal phase and the normal play
phase. Use `env.set_signals(signal_p0, signal_p1)` before calling `step()` to log cheap-talk messages. Signals
do not affect action legality or base rewards.

When `enable_challenges=True`, you may pass `challenge_p0`/`challenge_p1` to `step()` (or leave them unset). Each
challenge incurs `challenge_cost`. If the challenged player signaled an action different from the revealed action,
the challenged player pays `challenge_penalty`; otherwise, the challenger pays `challenge_penalty`.

Side bets, commitments, and noisy tells are logged when their corresponding config flags are enabled via the
`bet_*`, `commitment_*`, and `tell_*` fields in the event log. Their payoff rules are intentionally minimal and
can be layered on later without breaking the event schema.

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
