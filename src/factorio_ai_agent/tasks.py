"""Named task definitions for mock Factorio episodes."""

from __future__ import annotations

from dataclasses import dataclass


FREEPLAY_STARTING_INVENTORY: tuple[tuple[str, int], ...] = (
    ("burner_mining_drill", 1),
    ("stone_furnace", 1),
    ("iron_plate", 8),
)


@dataclass(frozen=True)
class TaskDefinition:
    """Static configuration for a mock Factorio task."""

    name: str
    description: str
    target_iron_plates: int
    max_steps: int
    require_burner_miner_for_success: bool = False
    starting_inventory: tuple[tuple[str, int], ...] = ()
    required_burner_mined_iron_ore: int | None = None


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
    "manual-first-plate": TaskDefinition(
        name="manual-first-plate",
        description="Produce one iron plate with manual ore mining allowed.",
        target_iron_plates=1,
        max_steps=50,
    ),
    "burner-first-plate": TaskDefinition(
        name="burner-first-plate",
        description="Produce one iron plate after building and fueling a burner miner.",
        target_iron_plates=1,
        max_steps=50,
        require_burner_miner_for_success=True,
    ),
    "burner-three-plates": TaskDefinition(
        name="burner-three-plates",
        description="Produce three iron plates through the burner-miner chain.",
        target_iron_plates=3,
        max_steps=80,
        require_burner_miner_for_success=True,
    ),
    "burner-ten-plates": TaskDefinition(
        name="burner-ten-plates",
        description="Produce ten iron plates through repeated burner-miner production.",
        target_iron_plates=10,
        max_steps=180,
        require_burner_miner_for_success=True,
    ),
    "freeplay-burner-first-plate": TaskDefinition(
        name="freeplay-burner-first-plate",
        description="Produce one additional iron plate from the Freeplay crashland inventory.",
        target_iron_plates=9,
        max_steps=50,
        require_burner_miner_for_success=True,
        starting_inventory=FREEPLAY_STARTING_INVENTORY,
        required_burner_mined_iron_ore=1,
    ),
    "freeplay-burner-three-plates": TaskDefinition(
        name="freeplay-burner-three-plates",
        description="Produce three additional iron plates from the Freeplay crashland inventory.",
        target_iron_plates=11,
        max_steps=80,
        require_burner_miner_for_success=True,
        starting_inventory=FREEPLAY_STARTING_INVENTORY,
        required_burner_mined_iron_ore=3,
    ),
    "freeplay-burner-ten-plates": TaskDefinition(
        name="freeplay-burner-ten-plates",
        description="Produce ten additional iron plates from the Freeplay crashland inventory.",
        target_iron_plates=18,
        max_steps=180,
        require_burner_miner_for_success=True,
        starting_inventory=FREEPLAY_STARTING_INVENTORY,
        required_burner_mined_iron_ore=10,
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
        require_burner_miner_for_success=task.require_burner_miner_for_success,
        starting_inventory=task.starting_inventory,
        required_burner_mined_iron_ore=task.required_burner_mined_iron_ore,
    )
