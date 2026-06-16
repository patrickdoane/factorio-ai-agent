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


def test_burner_progress_penalizes_manual_plate_shortcut() -> None:
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
