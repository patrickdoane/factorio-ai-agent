import pytest

from factorio_ai_agent.envs.mock_factorio_env import Action, MockFactorioEnv


def test_reset_returns_expected_observation_shape() -> None:
    env = MockFactorioEnv()

    observation, info = env.reset()

    assert info == {}
    assert observation["inventory"]["iron_ore"] == 0
    assert observation["inventory"]["coal"] == 0
    assert observation["inventory"]["stone"] == 0
    assert observation["placed_entities"]["stone_furnace"] == 0
    assert observation["production_state"]["miner_progress"] == 0
    assert observation["production_state"]["furnace_progress"] == 0
    assert observation["production_state"]["target_iron_plates"] == 1
    assert observation["production_state"]["burner_mined_iron_ore"] == 0
    assert observation["production_state"]["required_burner_mined_iron_ore"] == 0
    assert observation["step_count"] == 0
    assert observation["current_objective"] == "Craft stone furnace"


def test_reset_applies_configured_starting_inventory() -> None:
    env = MockFactorioEnv(
        starting_inventory={
            "burner_mining_drill": 1,
            "stone_furnace": 1,
            "iron_plate": 8,
        }
    )

    observation, _ = env.reset()

    assert observation["inventory"]["burner_mining_drill"] == 1
    assert observation["inventory"]["stone_furnace"] == 1
    assert observation["inventory"]["iron_plate"] == 8
    assert observation["placed_entities"]["burner_mining_drill"] == 0


def test_reset_rejects_unknown_starting_inventory_item() -> None:
    with pytest.raises(ValueError, match="Unknown starting inventory item"):
        MockFactorioEnv(starting_inventory={"copper_plate": 1})


def test_mining_and_invalid_craft_updates_state_and_reward() -> None:
    env = MockFactorioEnv()
    env.reset()

    observation, reward, terminated, truncated, info = env.step(Action.MINE_STONE.value)

    assert observation["inventory"]["stone"] == 1
    assert reward == -0.01
    assert not terminated
    assert not truncated
    assert info["valid_action"] is True

    observation, reward, terminated, truncated, info = env.step(
        Action.CRAFT_STONE_FURNACE.value
    )

    assert observation["inventory"]["stone"] == 1
    assert reward == pytest.approx(-0.06)
    assert not terminated
    assert not truncated
    assert info["valid_action"] is False


def test_truncates_when_max_steps_is_reached_without_success() -> None:
    env = MockFactorioEnv(max_steps=1)
    env.reset()

    observation, _, terminated, truncated, _ = env.step(Action.WAIT.value)

    assert observation["step_count"] == 1
    assert not terminated
    assert truncated


def test_action_names_and_valid_action_mask_are_exposed() -> None:
    env = MockFactorioEnv()
    env.reset()

    assert env.action_name(Action.MINE_STONE.value) == "MINE_STONE"
    assert env.action_names()[Action.WAIT.value] == "WAIT"

    mask = env.valid_action_mask()

    assert len(mask) == env.action_space.n
    assert mask[Action.MINE_STONE.value]
    assert mask[Action.WAIT.value]
    assert not mask[Action.CRAFT_STONE_FURNACE.value]

    np_mask = env.action_masks()

    assert np_mask.dtype == bool
    assert np_mask.tolist() == mask


def test_action_mask_allows_crafting_after_resources_are_available() -> None:
    env = MockFactorioEnv()
    env.reset()

    for _ in range(5):
        env.step(Action.MINE_STONE.value)

    mask = env.valid_action_mask()

    assert mask[Action.CRAFT_STONE_FURNACE.value]
    assert not mask[Action.CRAFT_BURNER_MINING_DRILL.value]

    env = MockFactorioEnv(
        starting_inventory={
            "stone_furnace": 1,
            "iron_plate": 3,
            "iron_gear_wheel": 3,
        }
    )
    env.reset()

    mask = env.valid_action_mask()

    assert mask[Action.CRAFT_BURNER_MINING_DRILL.value]
    assert mask[Action.CRAFT_IRON_GEAR_WHEEL.value]


def test_action_mask_allows_placement_and_fueling_when_ready() -> None:
    env = MockFactorioEnv(
        starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1}
    )
    env.reset()

    mask = env.valid_action_mask()

    assert mask[Action.PLACE_STONE_FURNACE.value]
    assert mask[Action.PLACE_BURNER_MINING_DRILL.value]
    assert not mask[Action.INSERT_COAL_FUEL.value]

    env.step(Action.PLACE_BURNER_MINING_DRILL.value)
    env.step(Action.MINE_COAL.value)

    mask = env.valid_action_mask()

    assert mask[Action.INSERT_COAL_FUEL.value]


def test_full_bootstrap_mask_caps_placed_furnaces() -> None:
    env = MockFactorioEnv(
        require_burner_miner_for_success=True,
        required_burner_mined_iron_ore=1,
        starting_inventory={"stone_furnace": 1, "stone": 5},
    )
    env.reset()

    env.step(Action.PLACE_STONE_FURNACE.value)
    mask = env.valid_action_mask()

    assert mask[Action.CRAFT_STONE_FURNACE.value]
    assert not mask[Action.PLACE_STONE_FURNACE.value]
    assert not mask[Action.WAIT.value]

    env.step(Action.CRAFT_STONE_FURNACE.value)
    mask = env.valid_action_mask()

    assert not mask[Action.MINE_STONE.value]
    assert not mask[Action.CRAFT_STONE_FURNACE.value]
    assert not mask[Action.PLACE_STONE_FURNACE.value]


def test_full_bootstrap_mask_allows_wait_only_for_active_production() -> None:
    env = MockFactorioEnv(
        require_burner_miner_for_success=True,
        required_burner_mined_iron_ore=1,
        starting_inventory={"stone_furnace": 1},
    )
    env.reset()

    assert not env.valid_action_mask()[Action.WAIT.value]

    env.step(Action.PLACE_STONE_FURNACE.value)
    assert not env.valid_action_mask()[Action.WAIT.value]

    env.step(Action.MINE_IRON_ORE.value)
    assert env.valid_action_mask()[Action.WAIT.value]


def test_unknown_action_name_raises_value_error() -> None:
    env = MockFactorioEnv()

    with pytest.raises(ValueError, match="Unknown action"):
        env.action_name(999)


def test_manual_sequence_produces_first_iron_plate() -> None:
    env = MockFactorioEnv()
    observation, _ = env.reset()

    actions = [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
        Action.WAIT.value,
    ]

    reward = 0.0
    terminated = False
    truncated = False
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)

    assert observation["inventory"]["iron_plate"] == 1
    assert reward > 9.0
    assert terminated
    assert not truncated


def test_curriculum_success_can_require_crafted_furnace() -> None:
    env = MockFactorioEnv(
        max_steps=12,
        target_iron_plates=0,
        success_condition="stone_furnace_crafted",
    )
    env.reset()

    terminated = False
    for action in [*[Action.MINE_STONE.value] * 5, Action.CRAFT_STONE_FURNACE.value]:
        observation, reward, terminated, truncated, _ = env.step(action)

    assert observation["inventory"]["stone_furnace"] == 1
    assert reward > 9.0
    assert terminated
    assert not truncated
    assert observation["current_objective"] == "Task complete"


def test_curriculum_success_can_require_crafted_burner_drill() -> None:
    env = MockFactorioEnv(
        max_steps=12,
        target_iron_plates=0,
        starting_inventory={"stone_furnace": 1, "iron_plate": 9},
        success_condition="burner_mining_drill_crafted",
    )
    env.reset()

    actions = [
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.CRAFT_IRON_GEAR_WHEEL.value,
        Action.CRAFT_BURNER_MINING_DRILL.value,
    ]
    terminated = False
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)

    assert observation["inventory"]["burner_mining_drill"] == 1
    assert reward > 9.0
    assert terminated
    assert not truncated


def test_curriculum_success_can_require_fueled_burner_drill() -> None:
    env = MockFactorioEnv(
        max_steps=10,
        target_iron_plates=0,
        starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
        success_condition="burner_mining_drill_fueled",
    )
    env.reset()

    actions = [
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
    ]
    terminated = False
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)

    assert observation["placed_entities"]["burner_mining_drill"] == 1
    assert observation["placed_entities"]["coal_fuel_inserted"] == 1
    assert reward > 9.0
    assert terminated
    assert not truncated


def test_manual_shortcut_does_not_complete_burner_required_task() -> None:
    env = MockFactorioEnv(max_steps=20, require_burner_miner_for_success=True)
    observation, _ = env.reset()

    actions = [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
        Action.WAIT.value,
    ]

    terminated = False
    truncated = False
    for action in actions:
        observation, _, terminated, truncated, _ = env.step(action)

    assert observation["inventory"]["iron_plate"] == 1
    assert observation["production_state"]["burner_mined_iron_ore"] == 0
    assert not terminated
    assert not truncated
    assert observation["current_objective"] == "Craft burner mining drill"


def test_burner_required_success_can_use_explicit_burner_ore_goal() -> None:
    env = MockFactorioEnv(
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
    observation, _ = env.reset()

    actions = [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.WAIT.value,
        Action.WAIT.value,
        Action.WAIT.value,
    ]

    terminated = False
    truncated = False
    for action in actions:
        observation, _, terminated, truncated, _ = env.step(action)

    assert observation["inventory"]["iron_plate"] == 9
    assert observation["production_state"]["burner_mined_iron_ore"] == 1
    assert terminated
    assert not truncated


def test_wait_advances_miner_and_furnace_production_timing() -> None:
    env = MockFactorioEnv(
        starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1}
    )
    observation, _ = env.reset()

    actions = [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
    ]
    for action in actions:
        observation, _, terminated, truncated, _ = env.step(action)

    assert observation["inventory"]["iron_plate"] == 0
    assert not terminated
    assert not truncated

    observation, reward, terminated, truncated, info = env.step(Action.WAIT.value)

    assert observation["production_state"]["miner_progress"] == 1
    assert observation["production_state"]["furnace_progress"] == 0
    assert observation["inventory"]["iron_plate"] == 0
    assert reward == pytest.approx(-0.01)
    assert not terminated
    assert "produced_iron_plate" not in info

    observation, reward, terminated, truncated, info = env.step(Action.WAIT.value)

    assert observation["inventory"]["iron_plate"] == 0
    assert observation["placed_entities"]["coal_fuel_inserted"] == 0
    assert observation["production_state"]["burner_mined_iron_ore"] == 1
    assert observation["production_state"]["miner_progress"] == 0
    assert observation["production_state"]["furnace_progress"] == 1
    assert reward == pytest.approx(-0.01)
    assert not terminated
    assert "produced_iron_plate" not in info

    observation, reward, terminated, truncated, info = env.step(Action.WAIT.value)

    assert observation["inventory"]["iron_plate"] == 1
    assert observation["placed_entities"]["coal_fuel_inserted"] == 0
    assert observation["production_state"]["burner_mined_iron_ore"] == 1
    assert observation["production_state"]["miner_progress"] == 0
    assert observation["production_state"]["furnace_progress"] == 0
    assert reward > 9.0
    assert terminated
    assert not truncated
    assert info["produced_iron_plate"] is True


def test_smelted_iron_plates_success_condition_focuses_objective() -> None:
    env = MockFactorioEnv(
        max_steps=10,
        target_iron_plates=1,
        starting_inventory={"stone_furnace": 1},
        success_condition="smelted_iron_plates",
    )
    observation, _ = env.reset()

    assert observation["current_objective"] == "Smelt 1 more iron plate"
    assert observation["success_condition"] == "smelted_iron_plates"
    assert env.valid_actions() == [Action.MINE_IRON_ORE.value, Action.WAIT.value, Action.PLACE_STONE_FURNACE.value]

    for action in [Action.PLACE_STONE_FURNACE.value, Action.MINE_IRON_ORE.value, Action.WAIT.value]:
        observation, _, terminated, truncated, _ = env.step(action)

    assert observation["current_objective"] == "Smelt 1 more iron plate"
    assert env.valid_actions() == [Action.MINE_IRON_ORE.value, Action.WAIT.value]
    assert not terminated
    assert not truncated

    observation, _, terminated, truncated, _ = env.step(Action.WAIT.value)

    assert observation["inventory"]["iron_plate"] == 1
    assert observation["current_objective"] == "Task complete"
    assert terminated
    assert not truncated


def test_smelted_iron_plates_rejects_off_task_actions() -> None:
    env = MockFactorioEnv(
        max_steps=10,
        target_iron_plates=9,
        starting_inventory={"stone_furnace": 1, "iron_plate": 2},
        success_condition="smelted_iron_plates",
    )
    observation, _ = env.reset()

    observation, reward, terminated, truncated, info = env.step(
        Action.CRAFT_IRON_GEAR_WHEEL.value
    )

    assert observation["inventory"]["iron_plate"] == 2
    assert reward == pytest.approx(-0.06)
    assert info["valid_action"] is False
    assert not terminated
    assert not truncated
