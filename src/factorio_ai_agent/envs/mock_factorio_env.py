"""A tiny deterministic Factorio-like Gymnasium environment."""

from __future__ import annotations

from copy import deepcopy
from enum import IntEnum
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(IntEnum):
    """Discrete actions available in the mock task."""

    MINE_IRON_ORE = 0
    MINE_COAL = 1
    MINE_STONE = 2
    CRAFT_STONE_FURNACE = 3
    CRAFT_BURNER_MINING_DRILL = 4
    PLACE_STONE_FURNACE = 5
    PLACE_BURNER_MINING_DRILL = 6
    INSERT_COAL_FUEL = 7
    WAIT = 8
    CRAFT_IRON_GEAR_WHEEL = 9


Inventory = dict[str, int]
PlacedEntities = dict[str, int]
Observation = dict[str, Any]
ProductionState = dict[str, int]
SuccessCondition = Literal[
    "iron_plates",
    "buffered_iron_plates",
    "smelted_iron_plates",
    "stone_furnace_crafted",
    "burner_mining_drill_crafted",
    "burner_mining_drill_fueled",
]


class MockFactorioEnv(gym.Env[Observation, int]):
    """Simulate a tiny resource-to-first-iron-plate Factorio task."""

    metadata = {"render_modes": ["ansi"]}

    max_steps: int

    inventory: Inventory

    placed_entities: PlacedEntities
    production_state: ProductionState
    step_count: int
    current_objective: str

    def __init__(
        self,
        max_steps: int = 100,
        target_iron_plates: int = 1,
        miner_ticks_per_ore: int = 2,
        furnace_ticks_per_plate: int = 2,
        require_burner_miner_for_success: bool = False,
        starting_inventory: Mapping[str, int] | None = None,
        required_burner_mined_iron_ore: int | None = None,
        success_condition: SuccessCondition = "iron_plates",
        use_furnace_output_buffer: bool = False,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.target_iron_plates = target_iron_plates
        self.miner_ticks_per_ore = miner_ticks_per_ore
        self.furnace_ticks_per_plate = furnace_ticks_per_plate
        self.require_burner_miner_for_success = require_burner_miner_for_success
        self.starting_inventory = dict(starting_inventory or {})
        self.success_condition = success_condition
        self.use_furnace_output_buffer = use_furnace_output_buffer
        self.required_burner_mined_iron_ore = (
            required_burner_mined_iron_ore
            if required_burner_mined_iron_ore is not None
            else target_iron_plates if require_burner_miner_for_success else 0
        )
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "inventory": spaces.Dict(
                    {
                        "iron_ore": spaces.Discrete(100),
                        "coal": spaces.Discrete(100),
                        "stone": spaces.Discrete(100),
                        "stone_furnace": spaces.Discrete(10),
                        "burner_mining_drill": spaces.Discrete(10),
                        "iron_plate": spaces.Discrete(100),
                        "iron_gear_wheel": spaces.Discrete(100),
                    }
                ),
                "placed_entities": spaces.Dict(
                    {
                        "stone_furnace": spaces.Discrete(10),
                        "burner_mining_drill": spaces.Discrete(10),
                        "coal_fuel_inserted": spaces.Discrete(10),
                    }
                ),
                "production_state": spaces.Dict(
                    {
                        "miner_progress": spaces.Discrete(miner_ticks_per_ore + 1),
                        "furnace_progress": spaces.Discrete(furnace_ticks_per_plate + 1),
                        "target_iron_plates": spaces.Discrete(100),
                        "burner_mined_iron_ore": spaces.Discrete(100),
                        "required_burner_mined_iron_ore": spaces.Discrete(100),
                        "furnace_output_iron_plate": spaces.Discrete(100),
                    }
                ),
                "step_count": spaces.Discrete(max_steps + 1),
                "current_objective": spaces.Text(max_length=80),
                "success_condition": spaces.Text(max_length=40),
            }
        )
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the task to its configured starting inventory."""
        super().reset(seed=seed)
        self.inventory = {
            "iron_ore": 0,
            "coal": 0,
            "stone": 0,
            "stone_furnace": 0,
            "burner_mining_drill": 0,
            "iron_plate": 0,
            "iron_gear_wheel": 0,
        }
        for item, amount in self.starting_inventory.items():
            if item not in self.inventory:
                raise ValueError(f"Unknown starting inventory item: {item}")
            self.inventory[item] = amount
        self.placed_entities = {
            "stone_furnace": 0,
            "burner_mining_drill": 0,
            "coal_fuel_inserted": 0,
        }
        self.production_state = {
            "miner_progress": 0,
            "furnace_progress": 0,
            "target_iron_plates": self.target_iron_plates,
            "burner_mined_iron_ore": 0,
            "required_burner_mined_iron_ore": self.required_burner_mined_iron_ore,
            "furnace_output_iron_plate": 0,
        }
        self.step_count = 0
        self.current_objective = self._objective()
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the Gymnasium step tuple."""
        if action not in self.action_names():
            raise ValueError(f"Unknown action: {action}")

        self.step_count += 1
        reward = -0.01
        info: dict[str, Any] = {"action": Action(action).name, "valid_action": True}

        if action not in self.valid_actions() or not self._apply_action(Action(action)):
            reward -= 0.05
            info["valid_action"] = False

        produced_plate = self._advance_production(Action(action))
        if produced_plate and self._produced_plate_count() <= self.target_iron_plates:
            reward += 10.0
            info["produced_iron_plate"] = True

        terminated = self._is_success()
        if terminated and not produced_plate:
            reward += 10.0
        self.current_objective = self._objective()
        truncated = self.step_count >= self.max_steps and not terminated
        return self._get_obs(), reward, terminated, truncated, info

    def valid_actions(self) -> list[int]:
        """Return actions that currently have an effect, plus WAIT."""
        if self.success_condition in {"buffered_iron_plates", "smelted_iron_plates"}:
            valid = [Action.MINE_IRON_ORE, Action.WAIT]
            if self.inventory["stone_furnace"] >= 1:
                valid.append(Action.PLACE_STONE_FURNACE)
            return [action.value for action in valid]

        valid = [
            Action.MINE_IRON_ORE,
        ]
        if self._can_mine_coal():
            valid.append(Action.MINE_COAL)
        if self._can_wait():
            valid.append(Action.WAIT)
        if self._can_mine_stone():
            valid.append(Action.MINE_STONE)
        if self.inventory["stone"] >= 5 and self._can_craft_stone_furnace():
            valid.append(Action.CRAFT_STONE_FURNACE)
        if self.inventory["iron_plate"] >= 2:
            valid.append(Action.CRAFT_IRON_GEAR_WHEEL)
        if (
            self.inventory["iron_plate"] >= 3
            and self.inventory["iron_gear_wheel"] >= 3
            and self.inventory["stone_furnace"] >= 1
        ):
            valid.append(Action.CRAFT_BURNER_MINING_DRILL)
        if self.inventory["stone_furnace"] >= 1 and self.placed_entities["stone_furnace"] == 0:
            valid.append(Action.PLACE_STONE_FURNACE)
        if self.inventory["burner_mining_drill"] >= 1:
            valid.append(Action.PLACE_BURNER_MINING_DRILL)
        if self.inventory["coal"] >= 1 and self.placed_entities["burner_mining_drill"] >= 1:
            valid.append(Action.INSERT_COAL_FUEL)
        return [action.value for action in valid]

    def _can_craft_stone_furnace(self) -> bool:
        if self.placed_entities["stone_furnace"] == 0:
            return True
        return (
            self.success_condition == "iron_plates"
            and self.require_burner_miner_for_success
            and self.inventory["stone_furnace"] == 0
            and self.inventory["burner_mining_drill"] == 0
            and self.placed_entities["burner_mining_drill"] == 0
        )

    def _can_mine_stone(self) -> bool:
        if not (
            self.success_condition == "iron_plates"
            and self.require_burner_miner_for_success
        ):
            return True
        if (
            self.inventory["burner_mining_drill"] > 0
            or self.placed_entities["burner_mining_drill"] > 0
        ):
            return False
        available_furnaces = (
            self.inventory["stone_furnace"] + self.placed_entities["stone_furnace"]
        )
        return available_furnaces < 2

    def _can_mine_coal(self) -> bool:
        if not (
            self.success_condition == "iron_plates"
            and self.require_burner_miner_for_success
        ):
            return True
        if (
            self.inventory["burner_mining_drill"] > 0
            or self.placed_entities["burner_mining_drill"] > 0
        ):
            return True
        available_fuel = (
            self.inventory["coal"] + self.placed_entities["coal_fuel_inserted"]
        )
        return available_fuel < self.required_burner_mined_iron_ore

    def _can_wait(self) -> bool:
        if not (
            self.success_condition == "iron_plates"
            and self.require_burner_miner_for_success
        ):
            return True
        furnace_can_advance = (
            self.placed_entities["stone_furnace"] > 0
            and self.inventory["iron_ore"] > 0
        )
        miner_can_advance = (
            self.placed_entities["burner_mining_drill"] > 0
            and self.placed_entities["coal_fuel_inserted"] > 0
        )
        return furnace_can_advance or miner_can_advance

    def valid_action_mask(self) -> list[bool]:
        """Return a boolean mask of actions that are currently valid."""
        valid = set(self.valid_actions())
        return [action.value in valid for action in Action]

    def action_masks(self) -> np.ndarray:
        """Return a NumPy action mask compatible with mask-aware RL libraries."""
        return np.array(self.valid_action_mask(), dtype=np.bool_)

    def action_names(self) -> dict[int, str]:
        """Return action ids mapped to stable action names."""
        return {action.value: action.name for action in Action}

    def action_name(self, action: int) -> str:
        """Return the stable name for a discrete action id."""
        try:
            return Action(action).name
        except ValueError as exc:
            raise ValueError(f"Unknown action: {action}") from exc

    def render(self) -> str:
        """Return a compact text rendering of the current state."""
        return (
            f"step={self.step_count} inventory={self.inventory} "
            f"placed={self.placed_entities} production={self.production_state} "
            f"objective={self.current_objective}"
        )

    def _apply_action(self, action: Action) -> bool:
        if action == Action.MINE_IRON_ORE:
            self.inventory["iron_ore"] += 1
            return True
        if action == Action.MINE_COAL:
            self.inventory["coal"] += 1
            return True
        if action == Action.MINE_STONE:
            self.inventory["stone"] += 1
            return True
        if action == Action.CRAFT_STONE_FURNACE:
            return self._craft({"stone": 5}, "stone_furnace")
        if action == Action.CRAFT_BURNER_MINING_DRILL:
            return self._craft(
                {"iron_plate": 3, "iron_gear_wheel": 3, "stone_furnace": 1},
                "burner_mining_drill",
            )
        if action == Action.CRAFT_IRON_GEAR_WHEEL:
            return self._craft({"iron_plate": 2}, "iron_gear_wheel")
        if action == Action.PLACE_STONE_FURNACE:
            return self._place("stone_furnace")
        if action == Action.PLACE_BURNER_MINING_DRILL:
            return self._place("burner_mining_drill")
        if action == Action.INSERT_COAL_FUEL:
            if self.inventory["coal"] < 1 or self.placed_entities["burner_mining_drill"] < 1:
                return False
            self.inventory["coal"] -= 1
            self.placed_entities["coal_fuel_inserted"] += 1
            return True
        return True

    def _craft(self, costs: Inventory, output: str) -> bool:
        if any(self.inventory[item] < amount for item, amount in costs.items()):
            return False
        for item, amount in costs.items():
            self.inventory[item] -= amount
        self.inventory[output] += 1
        return True

    def _place(self, entity: str) -> bool:
        if self.inventory[entity] < 1:
            return False
        self.inventory[entity] -= 1
        self.placed_entities[entity] += 1
        return True

    def _advance_production(self, action: Action) -> bool:
        if action != Action.WAIT:
            return False

        self._advance_miner()
        return self._advance_furnace()

    def _advance_miner(self) -> None:
        if (
            self.placed_entities["burner_mining_drill"] < 1
            or self.placed_entities["coal_fuel_inserted"] < 1
        ):
            self.production_state["miner_progress"] = 0
            return

        self.production_state["miner_progress"] += 1
        if self.production_state["miner_progress"] >= self.miner_ticks_per_ore:
            self.production_state["miner_progress"] = 0
            self.placed_entities["coal_fuel_inserted"] -= 1
            self.inventory["iron_ore"] += 1
            self.production_state["burner_mined_iron_ore"] += 1

    def _advance_furnace(self) -> bool:
        if self.placed_entities["stone_furnace"] < 1 or self.inventory["iron_ore"] < 1:
            self.production_state["furnace_progress"] = 0
            return False

        self.production_state["furnace_progress"] += 1
        if self.production_state["furnace_progress"] < self.furnace_ticks_per_plate:
            return False

        self.production_state["furnace_progress"] = 0
        self.inventory["iron_ore"] -= 1
        if self.use_furnace_output_buffer:
            self.production_state["furnace_output_iron_plate"] += 1
        else:
            self.inventory["iron_plate"] += 1
        return True

    def _produced_plate_count(self) -> int:
        if self.success_condition == "buffered_iron_plates":
            return self.production_state["furnace_output_iron_plate"]
        return self.inventory["iron_plate"]

    def _objective(self) -> str:
        if self._is_success():
            return "Task complete"
        if self.success_condition == "stone_furnace_crafted":
            return "Craft stone furnace"
        if self.success_condition == "buffered_iron_plates":
            remaining = self.target_iron_plates - self.production_state["furnace_output_iron_plate"]
            if remaining == 1:
                return "Smelt 1 more buffered iron plate"
            return f"Smelt {remaining} more buffered iron plates"
        if self.success_condition == "burner_mining_drill_crafted":
            if self.inventory["iron_gear_wheel"] < 3:
                return "Craft iron gear wheels"
            return "Craft burner mining drill"
        if self.success_condition == "burner_mining_drill_fueled":
            if self.placed_entities["burner_mining_drill"] < 1:
                return "Place burner mining drill"
            return "Fuel burner miner"
        if self.success_condition == "smelted_iron_plates":
            remaining = self.target_iron_plates - self.inventory["iron_plate"]
            if remaining == 1:
                return "Smelt 1 more iron plate"
            return f"Smelt {remaining} more iron plates"
        has_burner_miner = (
            self.inventory["burner_mining_drill"] >= 1
            or self.placed_entities["burner_mining_drill"] >= 1
        )
        if self.inventory["stone_furnace"] < 1 and self.placed_entities["stone_furnace"] < 1:
            return "Craft stone furnace"
        if (
            not has_burner_miner
            and self.inventory["iron_plate"] >= 2
            and self.inventory["iron_gear_wheel"] < 3
        ):
            return "Craft iron gear wheels"
        if self.inventory["burner_mining_drill"] < 1 and self.placed_entities["burner_mining_drill"] < 1:
            return "Craft burner mining drill"
        if self.placed_entities["stone_furnace"] < 1:
            return "Place stone furnace"
        if self.placed_entities["burner_mining_drill"] < 1:
            return "Place burner mining drill"
        if self.placed_entities["coal_fuel_inserted"] < 1:
            return "Fuel burner miner"
        if (
            self.require_burner_miner_for_success
            and self.production_state["burner_mined_iron_ore"]
            < self.required_burner_mined_iron_ore
        ):
            return "Produce ore with burner miner"
        remaining = self.target_iron_plates - self.inventory["iron_plate"]
        if remaining == 1:
            return "Produce 1 more iron plate"
        return f"Produce {remaining} more iron plates"

    def _is_success(self) -> bool:
        if self.success_condition == "stone_furnace_crafted":
            return (
                self.inventory["stone_furnace"] >= 1
                or self.placed_entities["stone_furnace"] >= 1
            )
        if self.success_condition == "burner_mining_drill_crafted":
            return (
                self.inventory["burner_mining_drill"] >= 1
                or self.placed_entities["burner_mining_drill"] >= 1
            )
        if self.success_condition == "burner_mining_drill_fueled":
            return (
                self.placed_entities["burner_mining_drill"] >= 1
                and self.placed_entities["coal_fuel_inserted"] >= 1
            )
        if self.success_condition == "buffered_iron_plates":
            return self.production_state["furnace_output_iron_plate"] >= self.target_iron_plates
        if self.inventory["iron_plate"] < self.target_iron_plates:
            return False
        if not self.require_burner_miner_for_success:
            return True
        return (
            self.production_state["burner_mined_iron_ore"]
            >= self.required_burner_mined_iron_ore
        )

    def _get_obs(self) -> Observation:
        return {
            "inventory": deepcopy(self.inventory),
            "placed_entities": deepcopy(self.placed_entities),
            "production_state": deepcopy(self.production_state),
            "step_count": self.step_count,
            "current_objective": self.current_objective,
            "success_condition": self.success_condition,
        }
