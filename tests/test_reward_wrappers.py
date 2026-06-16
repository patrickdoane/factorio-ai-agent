import pytest

from factorio_ai_agent.envs.mock_factorio_env import Action, MockFactorioEnv
from factorio_ai_agent.training.reward_wrappers import (
    BurnerProgressRewardWrapper,
    ProgressRewardWrapper,
)


def test_progress_reward_wrapper_rewards_useful_resources_until_cap() -> None:
    env = ProgressRewardWrapper(MockFactorioEnv(max_steps=20))
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.MINE_STONE))

    assert reward == 0.04
    assert info["progress_reward"] == 0.05

    for _ in range(7):
        env.step(int(Action.MINE_STONE))

    _, reward, _, _, info = env.step(int(Action.MINE_STONE))

    assert reward == -0.01
    assert "progress_reward" not in info


def test_progress_reward_wrapper_rewards_crafting_milestone() -> None:
    env = ProgressRewardWrapper(MockFactorioEnv(max_steps=20))
    env.reset(seed=1)
    for _ in range(5):
        env.step(int(Action.MINE_STONE))

    _, reward, _, _, info = env.step(int(Action.CRAFT_STONE_FURNACE))

    assert reward == 0.49
    assert info["progress_reward"] == 0.50


def test_progress_reward_wrapper_does_not_reward_repeated_milestone() -> None:
    env = ProgressRewardWrapper(MockFactorioEnv(max_steps=30))
    env.reset(seed=1)
    for _ in range(10):
        env.step(int(Action.MINE_STONE))

    _, reward, _, _, info = env.step(int(Action.CRAFT_STONE_FURNACE))
    assert reward == 0.49
    assert info["progress_reward"] == 0.50

    _, reward, _, _, info = env.step(int(Action.CRAFT_STONE_FURNACE))

    assert reward == -0.01
    assert "progress_reward" not in info


def test_burner_progress_allows_bootstrap_manual_plate_before_drill_exists() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=20, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    for action in [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(10.99)
    assert info["progress_reward"] == 1.0


def test_burner_progress_cancels_repeated_bootstrap_target_plate_reward() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=40, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    for action in [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(-0.01)
    assert info["progress_reward"] == -10.0


def test_burner_progress_caps_bootstrap_manual_plate_progress() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=80, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    for action in [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        Action.PLACE_STONE_FURNACE.value,
        *([Action.MINE_IRON_ORE.value, Action.WAIT.value, Action.WAIT.value] * 9),
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(-0.01)
    assert "progress_reward" not in info


def test_burner_progress_rewards_useful_bootstrap_gears() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=40, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    env.env.inventory["iron_plate"] = 2

    _, reward, _, _, info = env.step(int(Action.CRAFT_IRON_GEAR_WHEEL))

    assert reward == pytest.approx(2.99)
    assert info["progress_reward"] == 3.0


def test_burner_progress_rewards_bootstrap_second_furnace() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=40, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    env.env.inventory["stone"] = 5
    env.env.placed_entities["stone_furnace"] = 1
    env._max_resource_counts["stone"] = 5
    env._achieved_milestones.update({"stone_furnace", "placed_stone_furnace"})

    _, reward, _, _, info = env.step(int(Action.CRAFT_STONE_FURNACE))

    assert reward == pytest.approx(3.99)
    assert info["progress_reward"] == 4.0


def test_burner_progress_penalizes_extra_bootstrap_furnaces() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=40, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    env.env.inventory["stone"] = 5
    env.env.inventory["stone_furnace"] = 1
    env.env.placed_entities["stone_furnace"] = 1
    env._max_resource_counts["stone"] = 5
    env._achieved_milestones.update({"stone_furnace", "placed_stone_furnace"})

    _, reward, _, _, info = env.step(int(Action.CRAFT_STONE_FURNACE))

    assert reward == pytest.approx(-5.01)
    assert info["progress_reward"] == -5.0


def test_burner_progress_penalizes_extra_bootstrap_gears() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=80, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)
    for action in [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        Action.PLACE_STONE_FURNACE.value,
        *[Action.MINE_IRON_ORE.value, Action.WAIT.value, Action.WAIT.value] * 8,
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.CRAFT_IRON_GEAR_WHEEL.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.CRAFT_IRON_GEAR_WHEEL))

    assert reward == pytest.approx(-5.01)
    assert info["progress_reward"] == -5.0


def test_burner_progress_penalizes_placing_recipe_furnace_before_drill() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=0,
            starting_inventory={"stone_furnace": 1, "iron_plate": 9},
            success_condition="burner_mining_drill_crafted",
        )
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.PLACE_STONE_FURNACE))

    assert reward == pytest.approx(-10.01)
    assert info["progress_reward"] == -10.0


def test_burner_progress_rewards_coal_for_fuel_curriculum() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=0,
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
            success_condition="burner_mining_drill_fueled",
        )
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.MINE_COAL))

    assert reward == pytest.approx(6.04)
    assert info["progress_reward"] == 6.05


def test_burner_progress_does_not_repeatedly_reward_coal_for_fuel_curriculum() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=0,
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
            success_condition="burner_mining_drill_fueled",
        )
    )
    env.reset(seed=1)
    env.step(int(Action.MINE_COAL))

    _, reward, _, _, info = env.step(int(Action.MINE_COAL))

    assert reward == pytest.approx(-0.01)
    assert "progress_reward" not in info


def test_burner_progress_penalizes_furnace_placement_for_fuel_curriculum() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=0,
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
            success_condition="burner_mining_drill_fueled",
        )
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.PLACE_STONE_FURNACE))

    assert reward == pytest.approx(-8.01)
    assert info["progress_reward"] == -8.0


def test_burner_progress_penalizes_iron_for_fuel_curriculum() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=0,
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
            success_condition="burner_mining_drill_fueled",
        )
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.MINE_IRON_ORE))

    assert reward == pytest.approx(-3.96)
    assert info["progress_reward"] == pytest.approx(-3.95)


def test_burner_progress_penalizes_plate_for_fuel_curriculum() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=0,
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
            success_condition="burner_mining_drill_fueled",
        )
    )
    env.reset(seed=1)
    env.env.placed_entities["stone_furnace"] = 1
    env.env.inventory["iron_ore"] = 1
    env._achieved_milestones.add("placed_stone_furnace")
    env.step(int(Action.WAIT))

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(-6.01)
    assert info["progress_reward"] == -6.0


def test_burner_progress_penalizes_manual_plate_after_spending_starting_plates() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=9,
            require_burner_miner_for_success=True,
            required_burner_mined_iron_ore=1,
            starting_inventory={
                "burner_mining_drill": 1,
                "stone_furnace": 1,
                "iron_plate": 8,
            },
        )
    )
    env.reset(seed=1)
    for action in [
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(-0.01)
    assert info["progress_reward"] == -10.0


def test_burner_progress_rewards_burner_mined_ore() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=30,
            require_burner_miner_for_success=True,
            starting_inventory={"burner_mining_drill": 1},
        )
    )
    env.reset(seed=1)
    for action in [
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.WAIT.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(2.04)
    assert info["progress_reward"] == 2.05


def test_burner_progress_rewards_repeated_refueling_until_requirement_met() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=40,
            target_iron_plates=11,
            require_burner_miner_for_success=True,
            required_burner_mined_iron_ore=3,
            starting_inventory={
                "burner_mining_drill": 1,
                "stone_furnace": 1,
                "iron_plate": 8,
            },
        )
    )
    env.reset(seed=1)
    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.MINE_COAL.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.INSERT_COAL_FUEL))

    assert reward == pytest.approx(3.99)
    assert info["progress_reward"] == 4.0


def test_burner_progress_does_not_reward_excess_refueling() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=40,
            target_iron_plates=11,
            require_burner_miner_for_success=True,
            required_burner_mined_iron_ore=3,
            starting_inventory={
                "burner_mining_drill": 1,
                "stone_furnace": 1,
                "iron_plate": 8,
            },
        )
    )
    env.reset(seed=1)
    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.MINE_COAL.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.INSERT_COAL_FUEL))

    assert reward == pytest.approx(-0.01)
    assert "progress_reward" not in info


def test_burner_progress_penalizes_manual_ore_while_burner_ore_needed() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=40,
            target_iron_plates=11,
            require_burner_miner_for_success=True,
            required_burner_mined_iron_ore=3,
            starting_inventory={
                "burner_mining_drill": 1,
                "stone_furnace": 1,
                "iron_plate": 8,
            },
        )
    )
    env.reset(seed=1)
    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.WAIT.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.MINE_IRON_ORE))

    assert reward == pytest.approx(-2.01)
    assert info["progress_reward"] == pytest.approx(-2.0)


def test_burner_progress_allows_bootstrap_manual_ore_before_drill_exists() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=20, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.MINE_IRON_ORE))

    assert reward == pytest.approx(0.04)
    assert info["progress_reward"] == 0.05


def test_burner_progress_cancels_plate_reward_until_burner_requirement_met() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=40,
            target_iron_plates=11,
            require_burner_miner_for_success=True,
            required_burner_mined_iron_ore=3,
            starting_inventory={
                "burner_mining_drill": 1,
                "stone_furnace": 1,
                "iron_plate": 8,
            },
        )
    )
    env.reset(seed=1)
    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.MINE_IRON_ORE.value,
    ]:
        env.step(action)

    _, reward, _, _, info = env.step(int(Action.WAIT))

    assert reward == pytest.approx(-0.01)
    assert info["progress_reward"] == -10.0


def test_burner_progress_rewards_burner_chain_placement_milestones() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            require_burner_miner_for_success=True,
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
        )
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.PLACE_STONE_FURNACE))

    assert reward == pytest.approx(5.99)
    assert info["progress_reward"] == 6.0

    _, reward, _, _, info = env.step(int(Action.PLACE_BURNER_MINING_DRILL))

    assert reward == pytest.approx(2.99)
    assert info["progress_reward"] == 3.0


def test_burner_progress_penalizes_invalid_actions() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(max_steps=20, require_burner_miner_for_success=True)
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.INSERT_COAL_FUEL))

    assert reward == pytest.approx(-0.56)
    assert info["progress_reward"] == -0.5


def test_burner_progress_penalizes_unneeded_freeplay_gears() -> None:
    env = BurnerProgressRewardWrapper(
        MockFactorioEnv(
            max_steps=20,
            target_iron_plates=9,
            require_burner_miner_for_success=True,
            required_burner_mined_iron_ore=1,
            starting_inventory={
                "burner_mining_drill": 1,
                "stone_furnace": 1,
                "iron_plate": 8,
            },
        )
    )
    env.reset(seed=1)

    _, reward, _, _, info = env.step(int(Action.CRAFT_IRON_GEAR_WHEEL))

    assert reward == pytest.approx(-30.01)
    assert info["progress_reward"] == -30.0
