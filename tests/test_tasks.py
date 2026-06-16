import pytest

from factorio_ai_agent.tasks import get_task, resolve_task, task_names


def test_task_registry_contains_named_plate_tasks() -> None:
    assert task_names() == [
        "first-plate",
        "three-plates",
        "ten-plates",
        "manual-first-plate",
        "burner-first-plate",
        "burner-three-plates",
        "burner-ten-plates",
        "bootstrap-craft-furnace",
        "bootstrap-smelt-plates",
        "buffered-smelt-plate",
        "buffered-collect-plate",
        "buffered-collect-three-plates",
        "buffered-insert-smelt-plate",
        "buffered-insert-collect-plate",
        "buffered-insert-collect-three-plates",
        "buffered-miner-output-ore",
        "buffered-miner-collect-ore",
        "buffered-miner-transfer-plate",
        "bootstrap-craft-drill",
        "bootstrap-place-and-fuel-drill",
        "freeplay-burner-first-plate",
        "freeplay-burner-three-plates",
        "freeplay-burner-ten-plates",
    ]
    assert get_task("first-plate").target_iron_plates == 1
    assert get_task("three-plates").max_steps == 80
    assert not get_task("manual-first-plate").require_burner_miner_for_success
    assert get_task("burner-first-plate").require_burner_miner_for_success
    assert get_task("freeplay-burner-first-plate").starting_inventory == (
        ("burner_mining_drill", 1),
        ("stone_furnace", 1),
        ("iron_plate", 8),
    )
    assert get_task("freeplay-burner-first-plate").target_iron_plates == 9
    assert get_task("freeplay-burner-first-plate").required_burner_mined_iron_ore == 1
    assert get_task("bootstrap-craft-furnace").success_condition == "stone_furnace_crafted"
    assert get_task("bootstrap-smelt-plates").starting_inventory == (("stone_furnace", 1),)
    assert get_task("bootstrap-smelt-plates").success_condition == "smelted_iron_plates"
    assert get_task("buffered-smelt-plate").success_condition == "buffered_iron_plates"
    assert get_task("buffered-smelt-plate").use_furnace_output_buffer
    assert get_task("buffered-collect-plate").success_condition == "collected_iron_plates"
    assert get_task("buffered-collect-plate").use_furnace_output_buffer
    assert get_task("buffered-collect-three-plates").target_iron_plates == 3
    assert get_task("buffered-collect-three-plates").success_condition == "collected_iron_plates"
    assert get_task("buffered-collect-three-plates").use_furnace_output_buffer
    assert get_task("buffered-insert-smelt-plate").success_condition == "buffered_iron_plates"
    assert get_task("buffered-insert-smelt-plate").use_furnace_output_buffer
    assert get_task("buffered-insert-smelt-plate").use_furnace_input_buffer
    assert get_task("buffered-insert-collect-plate").success_condition == "collected_iron_plates"
    assert get_task("buffered-insert-collect-plate").use_furnace_output_buffer
    assert get_task("buffered-insert-collect-plate").use_furnace_input_buffer
    assert get_task("buffered-insert-collect-three-plates").target_iron_plates == 3
    assert get_task("buffered-insert-collect-three-plates").success_condition == "collected_iron_plates"
    assert get_task("buffered-insert-collect-three-plates").use_furnace_output_buffer
    assert get_task("buffered-insert-collect-three-plates").use_furnace_input_buffer
    assert get_task("buffered-miner-output-ore").success_condition == "buffered_iron_ore"
    assert get_task("buffered-miner-output-ore").use_miner_output_buffer
    assert get_task("buffered-miner-collect-ore").success_condition == "collected_iron_ore"
    assert get_task("buffered-miner-collect-ore").use_miner_output_buffer
    assert get_task("buffered-miner-transfer-plate").success_condition == "collected_iron_plates"
    assert get_task("buffered-miner-transfer-plate").use_furnace_output_buffer
    assert get_task("buffered-miner-transfer-plate").use_furnace_input_buffer
    assert get_task("buffered-miner-transfer-plate").use_miner_output_buffer
    assert get_task("bootstrap-craft-drill").success_condition == "burner_mining_drill_crafted"
    assert (
        get_task("bootstrap-place-and-fuel-drill").success_condition
        == "burner_mining_drill_fueled"
    )


def test_resolve_task_applies_overrides() -> None:
    task = resolve_task("first-plate", max_steps=123, target_iron_plates=7)

    assert task.name == "first-plate"
    assert task.max_steps == 123
    assert task.target_iron_plates == 7
    assert not task.require_burner_miner_for_success
    assert task.starting_inventory == ()
    assert task.success_condition == "iron_plates"
    assert not task.use_furnace_output_buffer
    assert not task.use_furnace_input_buffer
    assert not task.use_miner_output_buffer


def test_resolve_task_preserves_burner_requirement() -> None:
    task = resolve_task("burner-first-plate", max_steps=123, target_iron_plates=7)

    assert task.require_burner_miner_for_success


def test_resolve_task_preserves_freeplay_starting_inventory() -> None:
    task = resolve_task(
        "freeplay-burner-first-plate",
        max_steps=123,
        target_iron_plates=10,
    )

    assert task.starting_inventory == (
        ("burner_mining_drill", 1),
        ("stone_furnace", 1),
        ("iron_plate", 8),
    )
    assert task.required_burner_mined_iron_ore == 1


def test_get_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("unknown")
