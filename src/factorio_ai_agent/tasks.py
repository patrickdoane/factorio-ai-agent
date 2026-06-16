"""Named task definitions for mock Factorio episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SuccessCondition = Literal[
    "iron_plates",
    "buffered_iron_plates",
    "collected_iron_plates",
    "buffered_iron_ore",
    "collected_iron_ore",
    "smelted_iron_plates",
    "stone_furnace_crafted",
    "burner_mining_drill_crafted",
    "burner_mining_drill_fueled",
]


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
    success_condition: SuccessCondition = "iron_plates"
    use_furnace_output_buffer: bool = False
    use_furnace_input_buffer: bool = False
    use_miner_output_buffer: bool = False


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
        max_steps=80,
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
    "bootstrap-craft-furnace": TaskDefinition(
        name="bootstrap-craft-furnace",
        description="Mine enough stone and craft the first stone furnace from empty inventory.",
        target_iron_plates=0,
        max_steps=12,
        success_condition="stone_furnace_crafted",
    ),
    "bootstrap-smelt-plates": TaskDefinition(
        name="bootstrap-smelt-plates",
        description="Use a starter furnace to manually smelt plates for burner-drill crafting.",
        target_iron_plates=9,
        max_steps=40,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="smelted_iron_plates",
    ),
    "buffered-smelt-plate": TaskDefinition(
        name="buffered-smelt-plate",
        description="Smelt one iron plate into a furnace output buffer.",
        target_iron_plates=1,
        max_steps=10,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="buffered_iron_plates",
        use_furnace_output_buffer=True,
    ),
    "buffered-collect-plate": TaskDefinition(
        name="buffered-collect-plate",
        description="Smelt one iron plate into a furnace output buffer, then collect it.",
        target_iron_plates=1,
        max_steps=12,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="collected_iron_plates",
        use_furnace_output_buffer=True,
    ),
    "buffered-collect-three-plates": TaskDefinition(
        name="buffered-collect-three-plates",
        description="Repeatedly smelt and collect three plates through a furnace output buffer.",
        target_iron_plates=3,
        max_steps=24,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="collected_iron_plates",
        use_furnace_output_buffer=True,
    ),
    "buffered-insert-smelt-plate": TaskDefinition(
        name="buffered-insert-smelt-plate",
        description="Insert ore into a furnace input buffer and smelt one buffered plate.",
        target_iron_plates=1,
        max_steps=12,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="buffered_iron_plates",
        use_furnace_output_buffer=True,
        use_furnace_input_buffer=True,
    ),
    "buffered-insert-collect-plate": TaskDefinition(
        name="buffered-insert-collect-plate",
        description="Insert ore into a furnace input buffer, smelt one plate, and collect it.",
        target_iron_plates=1,
        max_steps=14,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="collected_iron_plates",
        use_furnace_output_buffer=True,
        use_furnace_input_buffer=True,
    ),
    "buffered-insert-collect-three-plates": TaskDefinition(
        name="buffered-insert-collect-three-plates",
        description="Repeatedly insert, smelt, and collect three plates through furnace buffers.",
        target_iron_plates=3,
        max_steps=30,
        starting_inventory=(("stone_furnace", 1),),
        success_condition="collected_iron_plates",
        use_furnace_output_buffer=True,
        use_furnace_input_buffer=True,
    ),
    "buffered-miner-output-ore": TaskDefinition(
        name="buffered-miner-output-ore",
        description="Produce one iron ore into a burner miner output buffer.",
        target_iron_plates=1,
        max_steps=8,
        starting_inventory=(("burner_mining_drill", 1),),
        success_condition="buffered_iron_ore",
        use_miner_output_buffer=True,
    ),
    "buffered-miner-collect-ore": TaskDefinition(
        name="buffered-miner-collect-ore",
        description="Produce one iron ore into a burner miner output buffer, then collect it.",
        target_iron_plates=1,
        max_steps=10,
        starting_inventory=(("burner_mining_drill", 1),),
        success_condition="collected_iron_ore",
        use_miner_output_buffer=True,
    ),
    "bootstrap-craft-drill": TaskDefinition(
        name="bootstrap-craft-drill",
        description="Craft gears and a burner mining drill from prepared plates and furnace.",
        target_iron_plates=0,
        max_steps=12,
        starting_inventory=(("stone_furnace", 1), ("iron_plate", 9)),
        success_condition="burner_mining_drill_crafted",
    ),
    "bootstrap-place-and-fuel-drill": TaskDefinition(
        name="bootstrap-place-and-fuel-drill",
        description="Place and fuel a prepared burner mining drill.",
        target_iron_plates=0,
        max_steps=10,
        starting_inventory=(("stone_furnace", 1), ("burner_mining_drill", 1)),
        success_condition="burner_mining_drill_fueled",
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
        success_condition=task.success_condition,
        use_furnace_output_buffer=task.use_furnace_output_buffer,
        use_furnace_input_buffer=task.use_furnace_input_buffer,
        use_miner_output_buffer=task.use_miner_output_buffer,
    )
