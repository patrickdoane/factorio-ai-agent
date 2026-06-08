import numpy as np

from factorio_ai_agent.envs.mock_factorio_env import Action, MockFactorioEnv
from factorio_ai_agent.envs.numeric_observation_wrapper import NumericObservationWrapper


def test_numeric_wrapper_returns_fixed_float_vector() -> None:
    env = NumericObservationWrapper(MockFactorioEnv())

    observation, _ = env.reset()

    assert observation.dtype == np.float32
    assert observation.shape == (13,)
    assert env.observation_space.contains(observation)


def test_numeric_wrapper_tracks_inventory_and_step_count() -> None:
    env = NumericObservationWrapper(MockFactorioEnv())
    env.reset()

    observation, _, _, _, _ = env.step(Action.MINE_STONE.value)

    assert observation.tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
