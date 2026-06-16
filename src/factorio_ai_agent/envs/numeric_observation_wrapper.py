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
    "iron_gear_wheel",
)
PLACED_ENTITY_KEYS = (
    "stone_furnace",
    "burner_mining_drill",
    "coal_fuel_inserted",
)
PRODUCTION_STATE_KEYS = (
    "miner_progress",
    "furnace_progress",
    "target_iron_plates",
    "burner_mined_iron_ore",
    "required_burner_mined_iron_ore",
    "furnace_output_iron_plate",
    "furnace_input_iron_ore",
    "miner_output_iron_ore",
)
SUCCESS_CONDITIONS = (
    "iron_plates",
    "buffered_iron_plates",
    "collected_iron_plates",
    "buffered_iron_ore",
    "collected_iron_ore",
    "smelted_iron_plates",
    "stone_furnace_crafted",
    "burner_mining_drill_crafted",
    "burner_mining_drill_fueled",
)


class NumericObservationWrapper(gym.ObservationWrapper[Observation, np.ndarray, int]):
    """Flatten mock dict observations into a numeric vector.

    The vector order is inventory counts, placed entity counts, production state,
    success-condition one-hot values, then step count. The current objective is
    intentionally excluded because it is a free-form string.
    """

    def __init__(self, env: gym.Env[Observation, int]) -> None:
        super().__init__(env)
        vector_size = (
            len(INVENTORY_KEYS)
            + len(PLACED_ENTITY_KEYS)
            + len(PRODUCTION_STATE_KEYS)
            + len(SUCCESS_CONDITIONS)
            + 1
        )
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
        values.extend(float(observation["production_state"][key]) for key in PRODUCTION_STATE_KEYS)
        success_condition = observation["success_condition"]
        values.extend(float(success_condition == condition) for condition in SUCCESS_CONDITIONS)
        values.append(float(observation["step_count"]))
        return np.array(values, dtype=np.float32)

    @property
    def unwrapped_env(self) -> Any:
        """Return the underlying environment for helper methods like action masks."""
        return self.env

    def action_masks(self) -> np.ndarray:
        """Return the current valid-action mask from the wrapped environment."""
        return self.env.action_masks()  # type: ignore[attr-defined]
