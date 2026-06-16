from factorio_ai_agent.envs.mock_factorio_env import Action, MockFactorioEnv
from factorio_ai_agent.training.reward_wrappers import ProgressRewardWrapper


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
    for _ in range(3):
        env.step(int(Action.MINE_IRON_ORE))
    for _ in range(3):
        env.step(int(Action.MINE_STONE))

    _, reward, _, _, info = env.step(int(Action.CRAFT_BURNER_MINING_DRILL))
    assert reward == 0.74
    assert info["progress_reward"] == 0.75

    env.step(int(Action.PLACE_BURNER_MINING_DRILL))
    for _ in range(3):
        env.step(int(Action.MINE_IRON_ORE))
    for _ in range(3):
        env.step(int(Action.MINE_STONE))

    _, reward, _, _, info = env.step(int(Action.CRAFT_BURNER_MINING_DRILL))

    assert reward == -0.01
    assert "progress_reward" not in info
