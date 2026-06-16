"""Training-only reward wrappers for learning experiments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from factorio_ai_agent.envs.mock_factorio_env import Action, MockFactorioEnv, Observation


class ProgressRewardWrapper(gym.Wrapper[Observation, int, Observation, int]):
    """Add dense training rewards for irreversible task progress.

    This wrapper is intentionally used only for training. Benchmarking and normal
    environment evaluation continue to use the unshaped environment rewards.
    """

    RESOURCE_TARGETS = {
        "stone": 8,
        "iron_ore": 3,
        "coal": 1,
    }
    RESOURCE_BONUS = 0.05
    MILESTONE_BONUSES = {
        "stone_furnace": 0.50,
        "burner_mining_drill": 0.75,
        "placed_stone_furnace": 0.50,
        "placed_burner_mining_drill": 0.75,
        "coal_fuel_inserted": 0.50,
        "iron_plate": 1.00,
    }

    def __init__(self, env: MockFactorioEnv) -> None:
        super().__init__(env)
        self._penalized_manual_plates = 0
        self._max_resource_counts = dict.fromkeys(self.RESOURCE_TARGETS, 0)
        self._achieved_milestones: set[str] = set()
        self._max_iron_plates = 0
        self._max_burner_mined_ore = 0
        self._started_with_burner_miner = False

    @property
    def unwrapped_env(self) -> MockFactorioEnv:
        return self.env

    def action_masks(self):  # type: ignore[no-untyped-def]
        return self.env.action_masks()

    def reset(self, **kwargs: Any) -> tuple[Observation, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._max_resource_counts = {
            resource: min(self.env.inventory[resource], target)
            for resource, target in self.RESOURCE_TARGETS.items()
        }
        self._achieved_milestones = set()
        if self.env.inventory["stone_furnace"] > 0:
            self._achieved_milestones.add("stone_furnace")
        if self.env.inventory["burner_mining_drill"] > 0:
            self._achieved_milestones.add("burner_mining_drill")
        self._max_iron_plates = self.env.inventory["iron_plate"]
        self._max_burner_mined_ore = self.env.production_state["burner_mined_iron_ore"]
        self._penalized_manual_plates = self.env.inventory["iron_plate"]
        self._started_with_burner_miner = self.env.inventory["burner_mining_drill"] > 0
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaping_reward = self._progress_reward()
        if shaping_reward:
            info["progress_reward"] = shaping_reward
        return observation, reward + shaping_reward, terminated, truncated, info

    def _progress_reward(self) -> float:
        reward = 0.0

        for resource, target in self.RESOURCE_TARGETS.items():
            previous_max = self._max_resource_counts[resource]
            current_count = min(self.env.inventory[resource], target)
            if current_count > previous_max:
                reward += (current_count - previous_max) * self.RESOURCE_BONUS
                self._max_resource_counts[resource] = current_count

        reward += self._first_milestone_reward(
            self.env.inventory["stone_furnace"] > 0,
            "stone_furnace",
        )
        reward += self._first_milestone_reward(
            self.env.inventory["burner_mining_drill"] > 0,
            "burner_mining_drill",
        )
        reward += self._first_milestone_reward(
            self.env.placed_entities["stone_furnace"] > 0,
            "placed_stone_furnace",
        )
        reward += self._first_milestone_reward(
            self.env.placed_entities["burner_mining_drill"] > 0,
            "placed_burner_mining_drill",
        )
        reward += self._first_milestone_reward(
            self.env.placed_entities["coal_fuel_inserted"] > 0,
            "coal_fuel_inserted",
        )
        if self.env.inventory["iron_plate"] > self._max_iron_plates:
            reward += (
                self.env.inventory["iron_plate"] - self._max_iron_plates
            ) * self.MILESTONE_BONUSES["iron_plate"]
            self._max_iron_plates = self.env.inventory["iron_plate"]
        return reward

    def _first_milestone_reward(
        self,
        achieved: bool,
        bonus_key: str,
    ) -> float:
        if not achieved or bonus_key in self._achieved_milestones:
            return 0.0
        self._achieved_milestones.add(bonus_key)
        return self.MILESTONE_BONUSES[bonus_key]


class BurnerProgressRewardWrapper(ProgressRewardWrapper):
    """Shape rewards for explicit burner-chain training tasks."""

    RESOURCE_TARGETS = {
        **ProgressRewardWrapper.RESOURCE_TARGETS,
        "stone": 10,
    }
    MILESTONE_BONUSES = {
        **ProgressRewardWrapper.MILESTONE_BONUSES,
        "burner_mining_drill": 8.00,
        "placed_stone_furnace": 6.00,
        "placed_burner_mining_drill": 3.00,
        "coal_fuel_inserted": 2.00,
    }
    BURNER_ORE_BONUS = 2.00
    BURNER_REFUEL_BONUS = 4.00
    FREEPLAY_GEAR_PENALTY = 30.00
    BOOTSTRAP_GEAR_BONUS = 3.00
    BOOTSTRAP_SECOND_FURNACE_BONUS = 4.00
    BOOTSTRAP_EXTRA_FURNACE_PENALTY = 5.00
    BOOTSTRAP_RECIPE_FURNACE_PLACEMENT_PENALTY = 10.00
    BOOTSTRAP_REQUIRED_PLATE_EQUIVALENT = 9
    BOOTSTRAP_EXTRA_GEAR_PENALTY = 5.00
    FUEL_TASK_COAL_BONUS = 6.00
    FUEL_TASK_IRON_DISTRACTION_PENALTY = 4.00
    FUEL_TASK_PLATE_DISTRACTION_PENALTY = 6.00
    FUEL_TASK_STONE_FURNACE_DISTRACTION_PENALTY = 8.00
    FURNACE_TASK_STONE_BONUS = 1.00
    FURNACE_TASK_EXCESS_STONE_PENALTY = 2.00
    FURNACE_TASK_RESOURCE_DISTRACTION_PENALTY = 2.00
    INVALID_ACTION_PENALTY = 0.50
    MANUAL_ORE_PENALTY = 2.05
    MANUAL_PLATE_REWARD = 10.00

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        previous_coal = self.env.inventory["coal"]
        previous_stone = self.env.inventory["stone"]
        observation, reward, terminated, truncated, info = self.env.step(action)
        previous_max_iron_plates = self._max_iron_plates
        valid_action = bool(info.get("valid_action"))
        shaping_reward = self._progress_reward()
        if self._needs_more_burner_ore() and valid_action:
            if action == Action.INSERT_COAL_FUEL.value and self._refuel_is_useful():
                shaping_reward += self.BURNER_REFUEL_BONUS
            if action == Action.MINE_IRON_ORE.value and not self._manual_bootstrap_allowed():
                shaping_reward -= self.MANUAL_ORE_PENALTY
        if self._produced_plate_before_burner_requirement(info):
            shaping_reward -= self.MANUAL_PLATE_REWARD
            if self.env.inventory["iron_plate"] > previous_max_iron_plates:
                shaping_reward -= self.MILESTONE_BONUSES["iron_plate"]
        if self._produced_repeated_bootstrap_target_plate(info):
            shaping_reward -= self.MANUAL_PLATE_REWARD
        if self._produced_excess_bootstrap_manual_plate(previous_max_iron_plates):
            shaping_reward -= self.MILESTONE_BONUSES["iron_plate"]
        if valid_action and self._crafted_useful_bootstrap_gear(action):
            shaping_reward += self.BOOTSTRAP_GEAR_BONUS
        if valid_action and self._crafted_bootstrap_second_furnace(action):
            shaping_reward += self.BOOTSTRAP_SECOND_FURNACE_BONUS
        if valid_action and self._crafted_extra_bootstrap_furnace(action):
            shaping_reward -= self.BOOTSTRAP_EXTRA_FURNACE_PENALTY
        if valid_action and self._placed_recipe_furnace_before_drill(action):
            shaping_reward -= (
                self.MILESTONE_BONUSES["placed_stone_furnace"]
                + self.BOOTSTRAP_RECIPE_FURNACE_PLACEMENT_PENALTY
            )
        if valid_action and self._fuel_task_mined_coal(action, previous_coal):
            shaping_reward += self.FUEL_TASK_COAL_BONUS
        if valid_action and self._fuel_task_placed_stone_furnace(action):
            shaping_reward -= (
                self.MILESTONE_BONUSES["placed_stone_furnace"]
                + self.FUEL_TASK_STONE_FURNACE_DISTRACTION_PENALTY
            )
        if valid_action and self._fuel_task_mined_iron(action):
            shaping_reward -= self.FUEL_TASK_IRON_DISTRACTION_PENALTY
        if self._fuel_task_produced_plate(previous_max_iron_plates):
            shaping_reward -= (
                self.MILESTONE_BONUSES["iron_plate"]
                + self.FUEL_TASK_PLATE_DISTRACTION_PENALTY
            )
        if valid_action and self._furnace_task_mined_useful_stone(action):
            shaping_reward += self.FURNACE_TASK_STONE_BONUS
        if valid_action and self._furnace_task_mined_excess_stone(action, previous_stone):
            shaping_reward -= self.FURNACE_TASK_EXCESS_STONE_PENALTY
        if valid_action and self._furnace_task_mined_distraction(action):
            shaping_reward -= self.FURNACE_TASK_RESOURCE_DISTRACTION_PENALTY
        if valid_action and self._crafted_unneeded_freeplay_gear(action):
            shaping_reward -= self.FREEPLAY_GEAR_PENALTY
        if valid_action and self._crafted_extra_bootstrap_gear(action):
            shaping_reward -= self.BOOTSTRAP_EXTRA_GEAR_PENALTY
        if not info["valid_action"]:
            shaping_reward -= self.INVALID_ACTION_PENALTY
        if shaping_reward:
            info["progress_reward"] = shaping_reward
        return observation, reward + shaping_reward, terminated, truncated, info

    def _progress_reward(self) -> float:
        reward = super()._progress_reward()

        if self.env.production_state["burner_mined_iron_ore"] > self._max_burner_mined_ore:
            reward += (
                self.env.production_state["burner_mined_iron_ore"]
                - self._max_burner_mined_ore
            ) * self.BURNER_ORE_BONUS
            self._max_burner_mined_ore = self.env.production_state["burner_mined_iron_ore"]

        return reward

    def _produced_plate_before_burner_requirement(self, info: dict[str, Any]) -> bool:
        return (
            self.env.require_burner_miner_for_success
            and bool(info.get("produced_iron_plate"))
            and self._needs_more_burner_ore()
            and not self._manual_bootstrap_allowed()
        )

    def _produced_repeated_bootstrap_target_plate(self, info: dict[str, Any]) -> bool:
        return (
            self._manual_bootstrap_allowed()
            and bool(info.get("produced_iron_plate"))
            and self.env.inventory["iron_plate"] <= self.env.target_iron_plates
            and self._bootstrap_manual_plate_equivalent() > self.env.target_iron_plates
        )

    def _produced_excess_bootstrap_manual_plate(self, previous_max_iron_plates: int) -> bool:
        return (
            self._manual_bootstrap_allowed()
            and self.env.inventory["iron_plate"] > previous_max_iron_plates
            and self._bootstrap_manual_plate_equivalent()
            > self.BOOTSTRAP_REQUIRED_PLATE_EQUIVALENT
        )

    def _needs_more_burner_ore(self) -> bool:
        return (
            self.env.require_burner_miner_for_success
            and self.env.production_state["burner_mined_iron_ore"]
            < self.env.required_burner_mined_iron_ore
        )

    def _refuel_is_useful(self) -> bool:
        remaining_required_ore = (
            self.env.required_burner_mined_iron_ore
            - self.env.production_state["burner_mined_iron_ore"]
        )
        return self.env.placed_entities["coal_fuel_inserted"] <= remaining_required_ore

    def _manual_bootstrap_allowed(self) -> bool:
        return (
            not self._started_with_burner_miner
            and self.env.inventory["burner_mining_drill"] == 0
            and self.env.placed_entities["burner_mining_drill"] == 0
        )

    def _crafted_unneeded_freeplay_gear(self, action: int) -> bool:
        return (
            self._started_with_burner_miner
            and self.env.require_burner_miner_for_success
            and action == Action.CRAFT_IRON_GEAR_WHEEL.value
            and not self._burner_task_is_successful()
        )

    def _crafted_useful_bootstrap_gear(self, action: int) -> bool:
        return (
            self._manual_bootstrap_allowed()
            and self.env.success_condition != "smelted_iron_plates"
            and action == Action.CRAFT_IRON_GEAR_WHEEL.value
            and self.env.inventory["iron_gear_wheel"] <= 3
        )

    def _crafted_bootstrap_second_furnace(self, action: int) -> bool:
        return (
            self._manual_bootstrap_allowed()
            and action == Action.CRAFT_STONE_FURNACE.value
            and self.env.success_condition == "iron_plates"
            and self.env.placed_entities["stone_furnace"] > 0
            and self.env.inventory["stone_furnace"] == 1
        )

    def _crafted_extra_bootstrap_furnace(self, action: int) -> bool:
        return (
            self._manual_bootstrap_allowed()
            and action == Action.CRAFT_STONE_FURNACE.value
            and self.env.placed_entities["stone_furnace"] > 0
            and self.env.inventory["stone_furnace"] > 1
        )

    def _placed_recipe_furnace_before_drill(self, action: int) -> bool:
        return (
            self.env.success_condition == "burner_mining_drill_crafted"
            and action == Action.PLACE_STONE_FURNACE.value
            and self.env.inventory["burner_mining_drill"] == 0
            and self.env.placed_entities["burner_mining_drill"] == 0
        )

    def _fuel_task_mined_coal(self, action: int, previous_coal: int) -> bool:
        return (
            self.env.success_condition == "burner_mining_drill_fueled"
            and action == Action.MINE_COAL.value
            and previous_coal == 0
            and self.env.inventory["coal"] > 0
        )

    def _fuel_task_placed_stone_furnace(self, action: int) -> bool:
        return (
            self.env.success_condition == "burner_mining_drill_fueled"
            and action == Action.PLACE_STONE_FURNACE.value
        )

    def _fuel_task_mined_iron(self, action: int) -> bool:
        return (
            self.env.success_condition == "burner_mining_drill_fueled"
            and action == Action.MINE_IRON_ORE.value
        )

    def _fuel_task_produced_plate(self, previous_max_iron_plates: int) -> bool:
        return (
            self.env.success_condition == "burner_mining_drill_fueled"
            and self.env.inventory["iron_plate"] > previous_max_iron_plates
        )

    def _furnace_task_mined_useful_stone(self, action: int) -> bool:
        return (
            self.env.success_condition == "stone_furnace_crafted"
            and action == Action.MINE_STONE.value
            and self.env.inventory["stone"] <= 5
        )

    def _furnace_task_mined_distraction(self, action: int) -> bool:
        return (
            self.env.success_condition == "stone_furnace_crafted"
            and action in {Action.MINE_COAL.value, Action.MINE_IRON_ORE.value}
        )

    def _furnace_task_mined_excess_stone(self, action: int, previous_stone: int) -> bool:
        return (
            self.env.success_condition == "stone_furnace_crafted"
            and action == Action.MINE_STONE.value
            and previous_stone >= 5
        )

    def _crafted_extra_bootstrap_gear(self, action: int) -> bool:
        return (
            self._manual_bootstrap_allowed()
            and action == Action.CRAFT_IRON_GEAR_WHEEL.value
            and self.env.inventory["iron_gear_wheel"] > 3
        )

    def _bootstrap_manual_plate_equivalent(self) -> int:
        return (
            self.env.inventory["iron_plate"]
            + self.env.inventory["iron_gear_wheel"] * 2
            + self.env.inventory["burner_mining_drill"] * 9
            + self.env.placed_entities["burner_mining_drill"] * 9
        )

    def _burner_task_is_successful(self) -> bool:
        return (
            self.env.inventory["iron_plate"] >= self.env.target_iron_plates
            and self.env.production_state["burner_mined_iron_ore"]
            >= self.env.required_burner_mined_iron_ore
        )
