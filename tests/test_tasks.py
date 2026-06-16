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


def test_resolve_task_applies_overrides() -> None:
    task = resolve_task("first-plate", max_steps=123, target_iron_plates=7)

    assert task.name == "first-plate"
    assert task.max_steps == 123
    assert task.target_iron_plates == 7
    assert not task.require_burner_miner_for_success
    assert task.starting_inventory == ()


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
