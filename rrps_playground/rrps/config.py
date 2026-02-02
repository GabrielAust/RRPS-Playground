"""Configuration for RRPS environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RRPSConfig:
    """Configuration for a RRPS game."""

    inventories_p0: Tuple[int, int, int] = (3, 3, 3)
    inventories_p1: Tuple[int, int, int] = (3, 3, 3)
    max_rounds: int | None = None
    history_len: int = 5
    seed: int | None = None

    include_self_counts: bool = True
    include_history: bool = True
    include_opponent_counts: bool = False

    reward_win: float = 1.0
    reward_tie: float = 0.0
    reward_loss: float = -1.0

    illegal_action_mode: str = "error"

    # Future toggles (placeholders only; not yet implemented)
    enable_signals: bool = False
    enable_challenges: bool = False
    enable_side_bets: bool = False
    enable_commitments: bool = False
    enable_noisy_tells: bool = False
    asymmetric_hands: bool = False
    deception_bonus: float = 0.0
    challenge_cost: float = 0.0
    challenge_penalty: float = 0.0

    def resolved_max_rounds(self) -> int:
        """Resolve max rounds from inventories when unset."""
        if self.max_rounds is not None:
            return self.max_rounds
        return int(sum(self.inventories_p0))

    def validate(self) -> None:
        """Validate configuration values."""
        if self.history_len < 0:
            raise ValueError("history_len must be >= 0")
        if self.illegal_action_mode not in {"error", "auto_mask_random", "forfeit_round"}:
            raise ValueError("invalid illegal_action_mode")
        if self.resolved_max_rounds() <= 0:
            raise ValueError("max_rounds must be positive")
        for counts in (self.inventories_p0, self.inventories_p1):
            if len(counts) != 3 or any(c < 0 for c in counts):
                raise ValueError("inventories must be length-3 non-negative tuples")
