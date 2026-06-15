"""Deterministic benchmark harness for autoresearch-style experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from factorio_ai_agent.agents.random_agent import RandomAgent
from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv, Observation
from factorio_ai_agent.tasks import TaskDefinition, get_task


@dataclass(frozen=True)
class BenchmarkEpisode:
    """One benchmark episode result."""

    task_name: str
    success: bool
    steps: int
    total_reward: float
    invalid_actions: int


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark result used by autonomous research loops."""

    score: float
    success_rate: float
    avg_steps: float
    avg_reward: float
    invalid_rate: float
    eval_episodes: int


def run_benchmark(
    *,
    agent_name: str,
    task_names: Iterable[str],
    eval_episodes: int,
    seed: int,
) -> BenchmarkSummary:
    """Run a deterministic benchmark over named tasks and return a summary."""
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1.")

    episodes: list[BenchmarkEpisode] = []
    for task_index, task_name in enumerate(task_names):
        task = get_task(task_name)
        for episode_index in range(eval_episodes):
            episode_seed = seed + task_index * eval_episodes + episode_index
            episodes.append(_run_episode(agent_name, task, episode_seed))

    return summarize_benchmark(episodes)


def summarize_benchmark(episodes: list[BenchmarkEpisode]) -> BenchmarkSummary:
    """Aggregate completed benchmark episodes into one comparable score."""
    if not episodes:
        raise ValueError("episodes must not be empty.")

    total_steps = sum(episode.steps for episode in episodes)
    total_invalid = sum(episode.invalid_actions for episode in episodes)
    success_rate = sum(episode.success for episode in episodes) / len(episodes)
    avg_steps = total_steps / len(episodes)
    avg_reward = sum(episode.total_reward for episode in episodes) / len(episodes)
    invalid_rate = total_invalid / total_steps if total_steps else 0.0
    score = success_rate - invalid_rate - avg_steps / 10_000.0

    return BenchmarkSummary(
        score=score,
        success_rate=success_rate,
        avg_steps=avg_steps,
        avg_reward=avg_reward,
        invalid_rate=invalid_rate,
        eval_episodes=len(episodes),
    )


def format_benchmark_summary(summary: BenchmarkSummary) -> str:
    """Format benchmark output as a stable machine-readable summary block."""
    return (
        "---\n"
        f"score:              {summary.score:.6f}\n"
        f"success_rate:       {summary.success_rate:.6f}\n"
        f"avg_steps:          {summary.avg_steps:.6f}\n"
        f"avg_reward:         {summary.avg_reward:.6f}\n"
        f"invalid_rate:       {summary.invalid_rate:.6f}\n"
        f"eval_episodes:      {summary.eval_episodes}"
    )


def parse_task_names(value: str) -> list[str]:
    """Parse a comma-separated task list for CLI use."""
    task_names = [task_name.strip() for task_name in value.split(",") if task_name.strip()]
    if not task_names:
        raise ValueError("At least one task name is required.")
    for task_name in task_names:
        get_task(task_name)
    return task_names


def _run_episode(agent_name: str, task: TaskDefinition, seed: int) -> BenchmarkEpisode:
    env = MockFactorioEnv(
        max_steps=task.max_steps,
        target_iron_plates=task.target_iron_plates,
    )
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    invalid_actions = 0

    random_agent = RandomAgent(seed=seed)
    scripted_agent = ScriptedBurnerAgent()

    while not terminated and not truncated:
        action = _select_action(agent_name, env, observation, random_agent, scripted_agent)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        invalid_actions += int(not info["valid_action"])

    return BenchmarkEpisode(
        task_name=task.name,
        success=terminated,
        steps=env.step_count,
        total_reward=total_reward,
        invalid_actions=invalid_actions,
    )


def _select_action(
    agent_name: str,
    env: MockFactorioEnv,
    observation: Observation,
    random_agent: RandomAgent,
    scripted_agent: ScriptedBurnerAgent,
) -> int:
    if agent_name == "random":
        return random_agent.act(env)
    if agent_name == "scripted":
        return scripted_agent.act(observation)
    raise ValueError("agent_name must be 'scripted' or 'random'.")
