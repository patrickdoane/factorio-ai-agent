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
    assert observation["step_count"] == 0
    assert observation["current_objective"] == "Mine resources"


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


def test_manual_sequence_produces_first_iron_plate() -> None:
    env = MockFactorioEnv()
    observation, _ = env.reset()

    actions = [
        *[Action.MINE_STONE.value] * 5,
        Action.CRAFT_STONE_FURNACE.value,
        *[Action.MINE_IRON_ORE.value] * 4,
        *[Action.MINE_STONE.value] * 3,
        Action.CRAFT_BURNER_MINING_DRILL.value,
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
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
