"""Named task definitions for mock Factorio episodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDefinition:
    """Static configuration for a mock Factorio task."""

    name: str
    description: str
    target_iron_plates: int
    max_steps: int


TASKS: dict[str, TaskDefinition] = {
    "first-plate": TaskDefinition(
        name="first-plate",
        description="Produce the first iron plate.",
        target_iron_plates=1,
        max_steps=50,
    ),
    "three-plates": TaskDefinition(
        name="three-plates",
        description="Produce three iron plates with simple production timing.",
        target_iron_plates=3,
        max_steps=80,
    ),
    "ten-plates": TaskDefinition(
        name="ten-plates",
        description="Produce ten iron plates as a longer repeated-production task.",
        target_iron_plates=10,
        max_steps=180,
    ),
}


def task_names() -> list[str]:
    """Return task names in a stable CLI-friendly order."""
    return list(TASKS)


def get_task(name: str) -> TaskDefinition:
    """Return a named task definition or raise a clear error."""
    try:
        return TASKS[name]
    except KeyError as exc:
        valid = ", ".join(task_names())
        raise ValueError(f"Unknown task {name!r}. Valid tasks: {valid}") from exc


def resolve_task(
    task_name: str,
    *,
    max_steps: int | None = None,
    target_iron_plates: int | None = None,
) -> TaskDefinition:
    """Return a task definition with optional CLI overrides applied."""
    task = get_task(task_name)
    if max_steps is None and target_iron_plates is None:
        return task
    return TaskDefinition(
        name=task.name,
        description=task.description,
        target_iron_plates=target_iron_plates or task.target_iron_plates,
        max_steps=max_steps or task.max_steps,
    )
