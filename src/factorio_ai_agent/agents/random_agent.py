"""Random baseline agent for the mock Factorio environment."""

from __future__ import annotations

import random
from typing import Protocol


class ValidActionEnv(Protocol):
    """Protocol for environments exposing valid discrete actions."""

    def valid_actions(self) -> list[int]: ...


class RandomAgent:
    """Choose uniformly from currently valid actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, env: ValidActionEnv) -> int:
        """Return a random valid action for the given environment."""
        return self._rng.choice(env.valid_actions())
