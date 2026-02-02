"""Utilities for storing and exporting gameplay transitions."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, List

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    info_small: Dict[str, object]


@dataclass
class TransitionBuffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, p0_transition: Transition, p1_transition: Transition) -> None:
        self.transitions.extend([p0_transition, p1_transition])

    def to_numpy(self) -> Dict[str, np.ndarray]:
        if not self.transitions:
            return {
                "obs": np.empty((0,)),
                "mask": np.empty((0,)),
                "action": np.empty((0,), dtype=np.int64),
                "reward": np.empty((0,), dtype=np.float32),
                "next_obs": np.empty((0,)),
                "done": np.empty((0,), dtype=bool),
                "info_small": np.empty((0,), dtype=object),
            }
        return {
            "obs": np.stack([transition.obs for transition in self.transitions]),
            "mask": np.stack([transition.mask for transition in self.transitions]),
            "action": np.array(
                [transition.action for transition in self.transitions], dtype=np.int64
            ),
            "reward": np.array(
                [transition.reward for transition in self.transitions],
                dtype=np.float32,
            ),
            "next_obs": np.stack(
                [transition.next_obs for transition in self.transitions]
            ),
            "done": np.array(
                [transition.done for transition in self.transitions], dtype=bool
            ),
            "info_small": np.array(
                [
                    json.dumps(transition.info_small, sort_keys=True)
                    for transition in self.transitions
                ],
                dtype=object,
            ),
        }

    def save_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for transition in self.transitions:
                record = {
                    "obs": transition.obs.tolist(),
                    "action": transition.action,
                    "reward": transition.reward,
                    "next_obs": transition.next_obs.tolist(),
                    "done": transition.done,
                    "mask": transition.mask.tolist(),
                    "info_small": transition.info_small,
                }
                handle.write(f"{json.dumps(record)}\n")

    def save_npz(self, path: str) -> None:
        """Save the buffer to a NumPy .npz archive."""
        data = self.to_numpy()
        np.savez_compressed(path, **data)
