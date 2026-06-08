"""Observation wrapper that flattens mock observations for RL libraries."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factorio_ai_agent.envs.mock_factorio_env import Observation


INVENTORY_KEYS = (
    "iron_ore",
    "coal",
    "stone",
    "stone_furnace",
    "burner_mining_drill",
    "iron_plate",
)
PLACED_ENTITY_KEYS = (
    "stone_furnace",
    "burner_mining_drill",
    "coal_fuel_inserted",
)


class NumericObservationWrapper(gym.ObservationWrapper[Observation, np.ndarray, int]):
    """Flatten mock dict observations into a numeric vector.

    The vector order is inventory counts, placed entity counts, then step count.
    The current objective is intentionally excluded for now because it is a string.
    """

    def __init__(self, env: gym.Env[Observation, int]) -> None:
        super().__init__(env)
        vector_size = len(INVENTORY_KEYS) + len(PLACED_ENTITY_KEYS) + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(vector_size,),
            dtype=np.float32,
        )

    def observation(self, observation: Observation) -> np.ndarray:
        """Convert a mock observation dictionary to a float32 vector."""
        values: list[float] = []
        values.extend(float(observation["inventory"][key]) for key in INVENTORY_KEYS)
        values.extend(float(observation["placed_entities"][key]) for key in PLACED_ENTITY_KEYS)
        values.append(float(observation["step_count"]))
        return np.array(values, dtype=np.float32)

    @property
    def unwrapped_env(self) -> Any:
        """Return the underlying environment for helper methods like action masks."""
        return self.env
