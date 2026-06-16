"""Small evaluation helpers for running agents over mock episodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv, Observation
from factorio_ai_agent.tasks import TaskDefinition, resolve_task


ActionSelector = Callable[[MockFactorioEnv, Observation], int]


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one completed or truncated episode."""

    success: bool
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    goal: str
    status: str

    @property
    def final_objective(self) -> str:
        """Backward-compatible alias for the final status label."""
        return self.status


def run_episode(
    select_action: ActionSelector,
    *,
    max_steps: int | None = None,
    quiet: bool = False,
    agent_name: str = "agent",
    episode_number: int = 1,
    seed: int | None = None,
    target_iron_plates: int | None = None,
    task_name: str = "first-plate",
) -> EpisodeResult:
    """Run one mock episode using a caller-provided action selector."""
    task = resolve_task(
        task_name,
        max_steps=max_steps,
        target_iron_plates=target_iron_plates,
    )
    env = _make_env(task)
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0

    if not quiet:
        print(format_episode_header(agent_name, episode_number))

    while not terminated and not truncated:
        action = select_action(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if not quiet:
            print(format_step(env, reward, info))

    result = EpisodeResult(
        success=terminated,
        steps=env.step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        goal=format_task_goal(task),
        status=observation["current_objective"],
    )
    print(format_result(result, episode_number))
    return result


def summarize_results(results: list[EpisodeResult]) -> dict[str, float]:
    """Summarize multiple episode results for quick agent comparisons."""
    if not results:
        return {"episodes": 0.0, "success_rate": 0.0, "avg_steps": 0.0, "avg_reward": 0.0}

    return {
        "episodes": float(len(results)),
        "success_rate": sum(result.success for result in results) / len(results),
        "avg_steps": sum(result.steps for result in results) / len(results),
        "avg_reward": sum(result.total_reward for result in results) / len(results),
    }


def _make_env(task: TaskDefinition) -> MockFactorioEnv:
    return MockFactorioEnv(
        max_steps=task.max_steps,
        target_iron_plates=task.target_iron_plates,
        require_burner_miner_for_success=task.require_burner_miner_for_success,
        starting_inventory=dict(task.starting_inventory),
        required_burner_mined_iron_ore=task.required_burner_mined_iron_ore,
        success_condition=task.success_condition,
        use_furnace_output_buffer=task.use_furnace_output_buffer,
    )


def format_summary(label: str, summary: dict[str, float]) -> str:
    """Return a readable multi-episode summary."""
    return (
        f"=== {label} Summary ===\n"
        f"Episodes: {summary['episodes']:.0f}\n"
        f"Success rate: {summary['success_rate'] * 100:.1f}%\n"
        f"Average steps: {summary['avg_steps']:.1f}\n"
        f"Average reward: {summary['avg_reward']:.2f}"
    )


def format_episode_header(agent_name: str, episode_number: int) -> str:
    return f"\n=== Episode {episode_number}: {agent_name} ==="


def format_step(env: MockFactorioEnv, reward: float, info: dict[str, object]) -> str:
    inventory = _format_counts(env.inventory)
    placed = _format_counts(env.placed_entities)
    production = _format_counts(env.production_state)
    return (
        f"\nStep {env.step_count:02d}\n"
        f"Action: {info['action']}\n"
        f"Valid action: {info['valid_action']}\n"
        f"Reward: {reward:.2f}\n"
        f"Objective: {env.current_objective}\n"
        f"Inventory: {inventory}\n"
        f"Placed: {placed}\n"
        f"Production: {production}"
    )


def format_result(result: EpisodeResult, episode_number: int) -> str:
    return (
        f"\n=== Episode {episode_number} Result ===\n"
        f"Success: {result.success}\n"
        f"Terminated: {result.terminated}\n"
        f"Truncated: {result.truncated}\n"
        f"Steps: {result.steps}\n"
        f"Total reward: {result.total_reward:.2f}\n"
        f"Goal: {result.goal}\n"
        f"Status: {result.status}"
    )


def _format_counts(values: dict[str, int]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items())


def format_goal(target_iron_plates: int) -> str:
    if target_iron_plates == 1:
        return "Produce 1 iron plate"
    return f"Produce {target_iron_plates} iron plates"


def format_task_goal(task: TaskDefinition) -> str:
    if task.success_condition == "stone_furnace_crafted":
        return "Craft stone furnace"
    if task.success_condition == "buffered_iron_plates":
        return format_goal(task.target_iron_plates)
    if task.success_condition == "collected_iron_plates":
        return format_goal(task.target_iron_plates)
    if task.success_condition == "smelted_iron_plates":
        return format_goal(task.target_iron_plates)
    if task.success_condition == "burner_mining_drill_crafted":
        return "Craft burner mining drill"
    if task.success_condition == "burner_mining_drill_fueled":
        return "Place and fuel burner mining drill"
    return format_goal(task.target_iron_plates)
