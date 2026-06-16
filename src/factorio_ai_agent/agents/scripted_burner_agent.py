"""Scripted agent that solves the first burner-miner mock task."""

from __future__ import annotations

from factorio_ai_agent.envs.mock_factorio_env import Action, Observation


class ScriptedBurnerAgent:
    """Follow a fixed resource, craft, place, fuel, and wait sequence."""

    def act(self, observation: Observation) -> int:
        """Return the next action needed to produce the target iron plates."""
        inventory = observation["inventory"]
        placed = observation["placed_entities"]
        production = observation["production_state"]
        target_iron_plates = production["target_iron_plates"]
        needs_burner_ore = (
            production["burner_mined_iron_ore"]
            < production["required_burner_mined_iron_ore"]
        )

        if inventory["stone_furnace"] < 1 and placed["stone_furnace"] < 1:
            if inventory["stone"] < 5:
                return Action.MINE_STONE.value
            return Action.CRAFT_STONE_FURNACE.value

        if placed["stone_furnace"] < 1:
            return Action.PLACE_STONE_FURNACE.value

        if needs_burner_ore and inventory["burner_mining_drill"] < 1 and placed["burner_mining_drill"] < 1:
            if inventory["iron_plate"] < 9:
                if inventory["iron_ore"] < 1:
                    return Action.MINE_IRON_ORE.value
                return Action.WAIT.value
            if inventory["iron_gear_wheel"] < 3:
                return Action.CRAFT_IRON_GEAR_WHEEL.value
            if inventory["stone_furnace"] < 1:
                if inventory["stone"] < 5:
                    return Action.MINE_STONE.value
                return Action.CRAFT_STONE_FURNACE.value
            return Action.CRAFT_BURNER_MINING_DRILL.value

        if placed["burner_mining_drill"] < 1:
            if inventory["burner_mining_drill"] >= 1:
                return Action.PLACE_BURNER_MINING_DRILL.value

        if inventory["iron_plate"] >= target_iron_plates and not needs_burner_ore:
            return Action.WAIT.value

        if needs_burner_ore and placed["coal_fuel_inserted"] < 1:
            if inventory["coal"] < 1:
                return Action.MINE_COAL.value
            return Action.INSERT_COAL_FUEL.value

        if inventory["iron_plate"] < target_iron_plates and inventory["iron_ore"] < 1:
            return Action.MINE_IRON_ORE.value

        return Action.WAIT.value
