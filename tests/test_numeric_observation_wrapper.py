import numpy as np

from factorio_ai_agent.envs.mock_factorio_env import Action, MockFactorioEnv
from factorio_ai_agent.envs.numeric_observation_wrapper import NumericObservationWrapper


def test_numeric_wrapper_returns_fixed_float_vector() -> None:
    env = NumericObservationWrapper(MockFactorioEnv())

    observation, _ = env.reset()

    assert observation.dtype == np.float32
    assert observation.shape == (29,)
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
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
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
    ]


def test_numeric_wrapper_encodes_success_condition() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(success_condition="burner_mining_drill_fueled")
    )

    observation, _ = env.reset()

    assert observation.tolist()[-10:-1] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_numeric_wrapper_distinguishes_buffered_plate_success_condition() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(success_condition="buffered_iron_plates")
    )

    observation, _ = env.reset()

    assert observation.tolist()[-10:-1] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_numeric_wrapper_distinguishes_collected_plate_success_condition() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(success_condition="collected_iron_plates")
    )

    observation, _ = env.reset()

    assert observation.tolist()[-10:-1] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_numeric_wrapper_distinguishes_buffered_ore_success_condition() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(success_condition="buffered_iron_ore")
    )

    observation, _ = env.reset()

    assert observation.tolist()[-10:-1] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_numeric_wrapper_distinguishes_collected_ore_success_condition() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(success_condition="collected_iron_ore")
    )

    observation, _ = env.reset()

    assert observation.tolist()[-10:-1] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_numeric_wrapper_distinguishes_smelted_plate_success_condition() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(success_condition="smelted_iron_plates")
    )

    observation, _ = env.reset()

    assert observation.tolist()[-10:-1] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_numeric_wrapper_exposes_furnace_output_buffer() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(
            starting_inventory={"stone_furnace": 1},
            success_condition="buffered_iron_plates",
            use_furnace_output_buffer=True,
        )
    )
    env.reset()

    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
        Action.WAIT.value,
    ]:
        env.step(action)
    observation, _, _, _, _ = env.step(Action.WAIT.value)

    assert observation.tolist()[16] == 1.0


def test_numeric_wrapper_exposes_furnace_input_buffer() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(
            starting_inventory={"stone_furnace": 1},
            success_condition="buffered_iron_plates",
            use_furnace_output_buffer=True,
            use_furnace_input_buffer=True,
        )
    )
    env.reset()

    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.MINE_IRON_ORE.value,
    ]:
        env.step(action)
    observation, _, _, _, _ = env.step(Action.INSERT_IRON_ORE_INTO_FURNACE.value)

    assert observation.tolist()[17] == 1.0


def test_numeric_wrapper_exposes_miner_output_buffer() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(
            starting_inventory={"burner_mining_drill": 1},
            success_condition="buffered_iron_ore",
            use_miner_output_buffer=True,
        )
    )
    env.reset()

    for action in [
        Action.PLACE_BURNER_MINING_DRILL.value,
        Action.MINE_COAL.value,
        Action.INSERT_COAL_FUEL.value,
        Action.WAIT.value,
    ]:
        env.step(action)
    observation, _, _, _, _ = env.step(Action.WAIT.value)

    assert observation.tolist()[18] == 1.0


def test_numeric_wrapper_exposes_miner_output_direction() -> None:
    env = NumericObservationWrapper(
        MockFactorioEnv(
            starting_inventory={"stone_furnace": 1, "burner_mining_drill": 1},
            success_condition="collected_iron_plates",
            use_furnace_output_buffer=True,
            use_furnace_input_buffer=True,
            use_miner_output_buffer=True,
            use_miner_output_direction=True,
        )
    )
    env.reset()

    for action in [
        Action.PLACE_STONE_FURNACE.value,
        Action.PLACE_BURNER_MINING_DRILL_OUTPUT_TO_FURNACE.value,
    ]:
        observation, _, _, _, _ = env.step(action)

    assert observation.tolist()[10] == 1.0


def test_numeric_wrapper_exposes_action_masks() -> None:
    env = NumericObservationWrapper(MockFactorioEnv())
    env.reset()

    mask = env.action_masks()

    assert mask.dtype == bool
    assert mask.shape == (env.action_space.n,)
    assert mask[Action.MINE_STONE.value]
    assert mask[Action.WAIT.value]
    assert not mask[Action.CRAFT_STONE_FURNACE.value]
