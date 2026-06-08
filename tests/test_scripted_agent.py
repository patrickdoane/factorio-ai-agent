from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv


def test_scripted_burner_agent_solves_mock_task() -> None:
    env = MockFactorioEnv(max_steps=50)
    agent = ScriptedBurnerAgent()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(observation)
        observation, _, terminated, truncated, _ = env.step(action)

    assert terminated
    assert not truncated
    assert observation["inventory"]["iron_plate"] == 1
    assert observation["current_objective"] == "Task complete"


def test_scripted_burner_agent_solves_repeated_plate_task() -> None:
    env = MockFactorioEnv(max_steps=80, target_iron_plates=3)
    agent = ScriptedBurnerAgent()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(observation)
        observation, _, terminated, truncated, _ = env.step(action)

    assert terminated
    assert not truncated
    assert observation["inventory"]["iron_plate"] == 3
    assert observation["current_objective"] == "Task complete"


def test_scripted_burner_agent_only_uses_valid_actions() -> None:
    env = MockFactorioEnv(max_steps=50)
    agent = ScriptedBurnerAgent()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(observation)
        assert action in env.valid_actions()
        observation, _, terminated, truncated, _ = env.step(action)
