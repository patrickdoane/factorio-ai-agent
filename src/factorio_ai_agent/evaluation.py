"""Small evaluation helpers for running agents over mock episodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv, Observation


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
    max_steps: int = 100,
    quiet: bool = False,
    agent_name: str = "agent",
    episode_number: int = 1,
    seed: int | None = None,
    target_iron_plates: int = 1,
) -> EpisodeResult:
    """Run one mock episode using a caller-provided action selector."""
    env = MockFactorioEnv(max_steps=max_steps, target_iron_plates=target_iron_plates)
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0

    if not quiet:
        print(_format_episode_header(agent_name, episode_number))

    while not terminated and not truncated:
        action = select_action(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if not quiet:
            print(_format_step(env, reward, info))

    result = EpisodeResult(
        success=terminated,
        steps=env.step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        goal=_format_goal(target_iron_plates),
        status=observation["current_objective"],
    )
    print(_format_result(result, episode_number))
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


def format_summary(label: str, summary: dict[str, float]) -> str:
    """Return a readable multi-episode summary."""
    return (
        f"=== {label} Summary ===\n"
        f"Episodes: {summary['episodes']:.0f}\n"
        f"Success rate: {summary['success_rate'] * 100:.1f}%\n"
        f"Average steps: {summary['avg_steps']:.1f}\n"
        f"Average reward: {summary['avg_reward']:.2f}"
    )


def _format_episode_header(agent_name: str, episode_number: int) -> str:
    return f"\n=== Episode {episode_number}: {agent_name} ==="


def _format_step(env: MockFactorioEnv, reward: float, info: dict[str, object]) -> str:
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


def _format_result(result: EpisodeResult, episode_number: int) -> str:
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


def _format_goal(target_iron_plates: int) -> str:
    if target_iron_plates == 1:
        return "Produce 1 iron plate"
    return f"Produce {target_iron_plates} iron plates"
