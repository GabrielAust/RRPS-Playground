"""RRPS environment implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import RRPSConfig
from .types import ObsDict, RewardTuple
from .utils import mask_from_counts, rps_outcome, sample_masked, update_counts


@dataclass
class RoundRecord:
    """Record of a single round."""

    round_index: int
    phase: str
    action_p0: int | None
    action_p1: int | None
    outcome_p0: int
    counts_p0: Tuple[int, int, int]
    counts_p1: Tuple[int, int, int]
    # Future toggles placeholders
    signal_p0: object | None = None
    signal_p1: object | None = None
    challenge_p0: object | None = None
    challenge_p1: object | None = None
    bet_p0: object | None = None
    bet_p1: object | None = None
    commitment_p0: object | None = None
    commitment_p1: object | None = None
    tell_p0: object | None = None
    tell_p1: object | None = None


class RRPSEnv:
    """RRPS environment with action masking and event logging."""

    def __init__(self, config: RRPSConfig | None = None) -> None:
        self.config = config or RRPSConfig()
        self.config.validate()
        self.max_rounds = self.config.resolved_max_rounds()
        self.rng = np.random.default_rng(self.config.seed)
        self.round_index = 0
        self.phase = "signal" if self.config.enable_signals else "play"
        self.counts_p0 = self.config.inventories_p0
        self.counts_p1 = self.config.inventories_p1
        self.history: List[Tuple[int, int]] = []
        self.events: List[Dict[str, object]] = []
        self.pre_step_hooks: List[Callable[["RRPSEnv"], None]] = []
        self.post_step_hooks: List[Callable[["RRPSEnv"], None]] = []
        self._pending_signal_p0: object | None = None
        self._pending_signal_p1: object | None = None
        self._pending_bet_p0: object | None = None
        self._pending_bet_p1: object | None = None
        self._pending_commitment_p0: object | None = None
        self._pending_commitment_p1: object | None = None
        self._pending_tell_p0: object | None = None
        self._pending_tell_p1: object | None = None

    def seed(self, seed: int | None) -> None:
        """Reseed RNG."""
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> ObsDict:
        """Reset environment and return initial observations."""
        if seed is not None:
            self.seed(seed)
        self.round_index = 0
        self.phase = "signal" if self.config.enable_signals else "play"
        self.counts_p0 = self.config.inventories_p0
        self.counts_p1 = self.config.inventories_p1
        self.history = []
        self.events = []
        self._pending_signal_p0 = None
        self._pending_signal_p1 = None
        self._pending_bet_p0 = None
        self._pending_bet_p1 = None
        self._pending_commitment_p0 = None
        self._pending_commitment_p1 = None
        self._pending_tell_p0 = None
        self._pending_tell_p1 = None
        return self._obs()

    def obs_dim(self) -> int:
        """Return observation dimension for a single player."""
        dim = 0
        if self.config.include_self_counts:
            dim += 3
        if self.config.include_opponent_counts:
            dim += 3
        if self.config.include_history:
            dim += self.config.history_len * 6
        dim += 1
        return dim

    def action_masks(self) -> Dict[str, np.ndarray]:
        """Return action masks for both players."""
        return {
            "p0": mask_from_counts(self.counts_p0),
            "p1": mask_from_counts(self.counts_p1),
        }

    def set_signals(self, signal_p0: object | None, signal_p1: object | None) -> None:
        """Record cheap-talk signals before stepping."""
        if not self.config.enable_signals:
            raise ValueError("Signals are disabled by config")
        self._pending_signal_p0 = signal_p0
        self._pending_signal_p1 = signal_p1
        self.phase = "play"

    def set_side_bets(self, bet_p0: object | None, bet_p1: object | None) -> None:
        """Record side bets before stepping."""
        if not self.config.enable_side_bets:
            raise ValueError("Side bets are disabled by config")
        self._pending_bet_p0 = bet_p0
        self._pending_bet_p1 = bet_p1

    def set_commitments(
        self, commitment_p0: object | None, commitment_p1: object | None
    ) -> None:
        """Record commitments before stepping."""
        if not self.config.enable_commitments:
            raise ValueError("Commitments are disabled by config")
        self._pending_commitment_p0 = commitment_p0
        self._pending_commitment_p1 = commitment_p1

    def set_noisy_tells(self, tell_p0: object | None, tell_p1: object | None) -> None:
        """Record noisy tells before stepping."""
        if not self.config.enable_noisy_tells:
            raise ValueError("Noisy tells are disabled by config")
        self._pending_tell_p0 = tell_p0
        self._pending_tell_p1 = tell_p1

    def step(
        self,
        action_p0: int,
        action_p1: int,
        *,
        signal_p0: object | None = None,
        signal_p1: object | None = None,
        challenge_p0: object | None = None,
        challenge_p1: object | None = None,
        bet_p0: object | None = None,
        bet_p1: object | None = None,
        commitment_p0: object | None = None,
        commitment_p1: object | None = None,
        tell_p0: object | None = None,
        tell_p1: object | None = None,
    ) -> Tuple[ObsDict, RewardTuple, bool, Dict[str, object]]:
        """Advance one step."""
        if self.is_terminated():
            raise ValueError("Episode already terminated")

        for hook in self.pre_step_hooks:
            hook(self)

        if self.config.enable_signals and self.phase == "signal":
            self.phase = "play"

        resolved_signal_p0, resolved_signal_p1 = self._resolve_pre_step_choice(
            signal_p0,
            signal_p1,
            self._pending_signal_p0,
            self._pending_signal_p1,
            self.config.enable_signals,
        )
        resolved_bet_p0, resolved_bet_p1 = self._resolve_pre_step_choice(
            bet_p0,
            bet_p1,
            self._pending_bet_p0,
            self._pending_bet_p1,
            self.config.enable_side_bets,
        )
        resolved_commitment_p0, resolved_commitment_p1 = self._resolve_pre_step_choice(
            commitment_p0,
            commitment_p1,
            self._pending_commitment_p0,
            self._pending_commitment_p1,
            self.config.enable_commitments,
        )
        resolved_tell_p0, resolved_tell_p1 = self._resolve_pre_step_choice(
            tell_p0,
            tell_p1,
            self._pending_tell_p0,
            self._pending_tell_p1,
            self.config.enable_noisy_tells,
        )
        self._pending_signal_p0 = None
        self._pending_signal_p1 = None
        self._pending_bet_p0 = None
        self._pending_bet_p1 = None
        self._pending_commitment_p0 = None
        self._pending_commitment_p1 = None
        self._pending_tell_p0 = None
        self._pending_tell_p1 = None

        mask_p0 = mask_from_counts(self.counts_p0)
        mask_p1 = mask_from_counts(self.counts_p1)

        resolved_p0 = self._resolve_action(action_p0, mask_p0, "p0")
        resolved_p1 = self._resolve_action(action_p1, mask_p1, "p1")

        outcome_p0 = 0
        if resolved_p0 is None and resolved_p1 is None:
            outcome_p0 = 0
            self.history.append((-1, -1))
        elif resolved_p0 is None:
            outcome_p0 = -1
            self.counts_p1 = update_counts(self.counts_p1, resolved_p1)
            self.history.append((-1, resolved_p1))
        elif resolved_p1 is None:
            outcome_p0 = 1
            self.counts_p0 = update_counts(self.counts_p0, resolved_p0)
            self.history.append((resolved_p0, -1))
        else:
            outcome_p0 = rps_outcome(resolved_p0, resolved_p1)
            self.counts_p0 = update_counts(self.counts_p0, resolved_p0)
            self.counts_p1 = update_counts(self.counts_p1, resolved_p1)
            self.history.append((resolved_p0, resolved_p1))

        rewards = list(self._rewards_from_outcome(outcome_p0))
        resolved_challenge_p0, resolved_challenge_p1 = self._resolve_challenges(
            challenge_p0, challenge_p1
        )
        if self.config.enable_challenges:
            self._apply_challenge_costs(
                rewards,
                resolved_signal_p0,
                resolved_signal_p1,
                resolved_p0,
                resolved_p1,
                resolved_challenge_p0,
                resolved_challenge_p1,
            )

        round_index = self.round_index
        event = RoundRecord(
            round_index=self.round_index,
            phase=self.phase,
            action_p0=resolved_p0,
            action_p1=resolved_p1,
            outcome_p0=outcome_p0,
            counts_p0=self.counts_p0,
            counts_p1=self.counts_p1,
            signal_p0=resolved_signal_p0,
            signal_p1=resolved_signal_p1,
            challenge_p0=resolved_challenge_p0,
            challenge_p1=resolved_challenge_p1,
            bet_p0=resolved_bet_p0,
            bet_p1=resolved_bet_p1,
            commitment_p0=resolved_commitment_p0,
            commitment_p1=resolved_commitment_p1,
            tell_p0=resolved_tell_p0,
            tell_p1=resolved_tell_p1,
        )
        self.events.append(event.__dict__)

        self.round_index += 1
        obs = self._obs()
        terminated = self.is_terminated()
        if self.config.enable_signals and not terminated:
            self.phase = "signal"
        info: Dict[str, object] = {
            "events_tail": self.events[-1:],
            "action_mask": self.action_masks(),
            "round": round_index,
            "phase": self.phase,
            "enable_signals": self.config.enable_signals,
            "enable_challenges": self.config.enable_challenges,
            "enable_side_bets": self.config.enable_side_bets,
            "enable_commitments": self.config.enable_commitments,
            "enable_noisy_tells": self.config.enable_noisy_tells,
        }
        for hook in self.post_step_hooks:
            hook(self)
        return obs, (float(rewards[0]), float(rewards[1])), terminated, info

    def is_terminated(self) -> bool:
        """Check termination condition."""
        if self.round_index >= self.max_rounds:
            return True
        no_moves = all(c == 0 for c in self.counts_p0) and all(
            c == 0 for c in self.counts_p1
        )
        return no_moves

    def _resolve_action(self, action: int, mask: np.ndarray, player: str) -> int | None:
        if action in (0, 1, 2) and mask[action] == 1:
            return int(action)

        mode = self.config.illegal_action_mode
        if mode == "error":
            raise ValueError(f"Illegal action {action} for {player}")
        if mode == "auto_mask_random":
            if mask.sum() == 0:
                return None
            return sample_masked(self.rng, mask)
        if mode == "forfeit_round":
            return None
        raise ValueError("Unsupported illegal_action_mode")

    def _rewards_from_outcome(self, outcome_p0: int) -> RewardTuple:
        if outcome_p0 == 1:
            return (self.config.reward_win, self.config.reward_loss)
        if outcome_p0 == -1:
            return (self.config.reward_loss, self.config.reward_win)
        return (self.config.reward_tie, self.config.reward_tie)

    @staticmethod
    def _resolve_pre_step_choice(
        provided_p0: object | None,
        provided_p1: object | None,
        pending_p0: object | None,
        pending_p1: object | None,
        enabled: bool,
    ) -> Tuple[object | None, object | None]:
        if not enabled:
            return None, None
        resolved_p0 = provided_p0 if provided_p0 is not None else pending_p0
        resolved_p1 = provided_p1 if provided_p1 is not None else pending_p1
        return resolved_p0, resolved_p1

    def _resolve_challenges(
        self, challenge_p0: object | None, challenge_p1: object | None
    ) -> Tuple[object | None, object | None]:
        if not self.config.enable_challenges:
            return None, None
        return challenge_p0, challenge_p1

    def _apply_challenge_costs(
        self,
        rewards: List[float],
        signal_p0: object | None,
        signal_p1: object | None,
        action_p0: int | None,
        action_p1: int | None,
        challenge_p0: object | None,
        challenge_p1: object | None,
    ) -> None:
        if challenge_p0:
            rewards[0] -= self.config.challenge_cost
            if self._challenge_correct(signal_p1, action_p1):
                rewards[1] -= self.config.challenge_penalty
            else:
                rewards[0] -= self.config.challenge_penalty
        if challenge_p1:
            rewards[1] -= self.config.challenge_cost
            if self._challenge_correct(signal_p0, action_p0):
                rewards[0] -= self.config.challenge_penalty
            else:
                rewards[1] -= self.config.challenge_penalty

    @staticmethod
    def _challenge_correct(signal: object | None, action: int | None) -> bool:
        if signal in (0, 1, 2):
            if action is None:
                return True
            return int(action) != int(signal)
        return False

    def _obs(self) -> ObsDict:
        p0 = self._build_obs(player="p0")
        p1 = self._build_obs(player="p1")
        return {"p0": p0, "p1": p1}

    def _build_obs(self, player: str) -> np.ndarray:
        features: List[float] = []
        counts_self = self.counts_p0 if player == "p0" else self.counts_p1
        counts_opp = self.counts_p1 if player == "p0" else self.counts_p0
        if self.config.include_self_counts:
            features.extend(float(c) for c in counts_self)
        if self.config.include_opponent_counts:
            features.extend(float(c) for c in counts_opp)
        if self.config.include_history:
            history = self.history[-self.config.history_len :]
            padded = [(-1, -1)] * (self.config.history_len - len(history)) + history
            for act_self, act_opp in padded:
                one_hot_self = [0.0, 0.0, 0.0]
                one_hot_opp = [0.0, 0.0, 0.0]
                if player == "p0":
                    self_action, opp_action = act_self, act_opp
                else:
                    self_action, opp_action = act_opp, act_self
                if self_action != -1:
                    one_hot_self[self_action] = 1.0
                if opp_action != -1:
                    one_hot_opp[opp_action] = 1.0
                features.extend(one_hot_self + one_hot_opp)
        features.append(self.round_index / float(self.max_rounds))
        return np.asarray(features, dtype=np.float32)
