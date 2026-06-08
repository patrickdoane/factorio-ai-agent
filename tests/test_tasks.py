import pytest

from factorio_ai_agent.tasks import get_task, resolve_task, task_names


def test_task_registry_contains_named_plate_tasks() -> None:
    assert task_names() == ["first-plate", "three-plates", "ten-plates"]
    assert get_task("first-plate").target_iron_plates == 1
    assert get_task("three-plates").max_steps == 80


def test_resolve_task_applies_overrides() -> None:
    task = resolve_task("first-plate", max_steps=123, target_iron_plates=7)

    assert task.name == "first-plate"
    assert task.max_steps == 123
    assert task.target_iron_plates == 7


def test_get_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("unknown")
