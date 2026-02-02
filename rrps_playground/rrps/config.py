"""Configuration for RRPS environment."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple


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


def apply_config_overrides(
    config: RRPSConfig, overrides: Mapping[str, Any]
) -> RRPSConfig:
    """Return a new config with overrides applied."""
    allowed = {field.name for field in fields(RRPSConfig)}
    updates: dict[str, Any] = {}
    counts = overrides.get("counts")
    if counts is not None and "inventories_p0" not in overrides:
        updates["inventories_p0"] = _normalize_counts(counts, "counts")
    if counts is not None and "inventories_p1" not in overrides:
        updates["inventories_p1"] = _normalize_counts(counts, "counts")
    for key, value in overrides.items():
        if key == "counts":
            continue
        if key not in allowed:
            raise ValueError(f"unknown config key: {key}")
        if key in {"inventories_p0", "inventories_p1"}:
            value = _normalize_counts(value, key)
        updates[key] = value
    return replace(config, **updates)


def load_config_file(path: str | Path) -> Mapping[str, Any]:
    """Load a config file from JSON or YAML."""
    path = Path(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif path.suffix in {".yaml", ".yml"}:
        data = _load_yaml(path)
    else:
        raise ValueError("config file must be .json, .yaml, or .yml")
    if not isinstance(data, Mapping):
        raise ValueError("config file must contain a JSON/YAML object")
    return data


def find_default_config(candidates: Iterable[str] | None = None) -> Path | None:
    """Find the first config file that exists in the current working directory."""
    if candidates is None:
        candidates = ("config.json", "config.yaml", "config.yml")
    for candidate in candidates:
        path = Path.cwd() / candidate
        if path.exists():
            return path
    return None


def _normalize_counts(value: Any, name: str) -> Tuple[int, int, int]:
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(item) for item in value)
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(item) for item in value)
    raise ValueError(f"{name} must be a length-3 list/tuple of ints")


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if importlib.util.find_spec("yaml") is None:
        raise ValueError("PyYAML is required to load YAML configs")
    yaml = importlib.import_module("yaml")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("YAML config must contain a mapping")
    return data
