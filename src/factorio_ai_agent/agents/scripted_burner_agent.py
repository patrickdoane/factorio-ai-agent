"""Scripted agent that solves the first burner-miner mock task."""

from __future__ import annotations

from factorio_ai_agent.envs.mock_factorio_env import Action, Observation


class ScriptedBurnerAgent:
    """Follow a fixed resource, craft, place, fuel, and wait sequence."""

    def act(self, observation: Observation) -> int:
        """Return the next action needed to produce the first iron plate."""
        inventory = observation["inventory"]
        placed = observation["placed_entities"]

        if inventory["stone_furnace"] < 1 and placed["stone_furnace"] < 1:
            if inventory["stone"] < 5:
                return Action.MINE_STONE.value
            return Action.CRAFT_STONE_FURNACE.value

        if inventory["burner_mining_drill"] < 1 and placed["burner_mining_drill"] < 1:
            if inventory["iron_ore"] < 4:
                return Action.MINE_IRON_ORE.value
            if inventory["stone"] < 3:
                return Action.MINE_STONE.value
            return Action.CRAFT_BURNER_MINING_DRILL.value

        if placed["stone_furnace"] < 1:
            return Action.PLACE_STONE_FURNACE.value

        if placed["burner_mining_drill"] < 1:
            return Action.PLACE_BURNER_MINING_DRILL.value

        if placed["coal_fuel_inserted"] < 1:
            if inventory["coal"] < 1:
                return Action.MINE_COAL.value
            return Action.INSERT_COAL_FUEL.value

        return Action.WAIT.value
