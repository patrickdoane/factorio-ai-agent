"""Deterministic benchmark harness for autoresearch-style experiments."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from factorio_ai_agent.agents.random_agent import RandomAgent
from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv, Observation
from factorio_ai_agent.envs.numeric_observation_wrapper import NumericObservationWrapper
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
    model_path: str | Path | None = None,
    model_algo: str = "ppo",
) -> BenchmarkSummary:
    """Run a deterministic benchmark over named tasks and return a summary."""
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1.")
    ppo_model = _load_ppo_model(model_path, model_algo=model_algo) if agent_name == "ppo" else None

    episodes: list[BenchmarkEpisode] = []
    for task_index, task_name in enumerate(task_names):
        task = get_task(task_name)
        for episode_index in range(eval_episodes):
            episode_seed = seed + task_index * eval_episodes + episode_index
            episodes.append(_run_episode(agent_name, task, episode_seed, ppo_model))

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


def append_results_tsv(
    path: str | Path,
    *,
    summary: BenchmarkSummary,
    agent_name: str,
    task_names: Iterable[str],
    seed: int,
) -> Path:
    """Append one benchmark summary row to a TSV log file."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not result_path.exists() or result_path.stat().st_size == 0

    with result_path.open("a", encoding="utf-8") as file:
        if should_write_header:
            file.write(_results_header() + "\n")
        file.write(_results_row(summary, agent_name, task_names, seed) + "\n")

    return result_path


def parse_task_names(value: str) -> list[str]:
    """Parse a comma-separated task list for CLI use."""
    task_names = [task_name.strip() for task_name in value.split(",") if task_name.strip()]
    if not task_names:
        raise ValueError("At least one task name is required.")
    for task_name in task_names:
        get_task(task_name)
    return task_names


def _results_header() -> str:
    return "\t".join(
        [
            "timestamp",
            "git_commit",
            "agent",
            "tasks",
            "seed",
            "score",
            "success_rate",
            "avg_steps",
            "avg_reward",
            "invalid_rate",
            "eval_episodes",
        ]
    )


def _results_row(
    summary: BenchmarkSummary,
    agent_name: str,
    task_names: Iterable[str],
    seed: int,
) -> str:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    return "\t".join(
        [
            timestamp,
            _git_commit(),
            agent_name,
            ",".join(task_names),
            str(seed),
            f"{summary.score:.6f}",
            f"{summary.success_rate:.6f}",
            f"{summary.avg_steps:.6f}",
            f"{summary.avg_reward:.6f}",
            f"{summary.invalid_rate:.6f}",
            str(summary.eval_episodes),
        ]
    )


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _load_ppo_model(model_path: str | Path | None, *, model_algo: str = "ppo") -> object:
    if model_path is None:
        raise ValueError("model_path is required when agent_name is 'ppo'.")
    if model_algo not in {"ppo", "maskable-ppo"}:
        raise ValueError("model_algo must be 'ppo' or 'maskable-ppo'.")

    if model_algo == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise ImportError(
                "Stable-Baselines3 is required to benchmark PPO models. "
                "Install with: pip install -e '.[rl]'"
            ) from error
        return PPO.load(model_path, device="cpu")

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise ImportError(
            "sb3-contrib is required to benchmark MaskablePPO models. "
            "Install with: pip install -e '.[rl]'"
        ) from error
    return MaskablePPO.load(model_path, device="cpu")


def _make_mock_env(task: TaskDefinition) -> MockFactorioEnv:
    return MockFactorioEnv(
        max_steps=task.max_steps,
        target_iron_plates=task.target_iron_plates,
        require_burner_miner_for_success=task.require_burner_miner_for_success,
        starting_inventory=dict(task.starting_inventory),
        required_burner_mined_iron_ore=task.required_burner_mined_iron_ore,
        success_condition=task.success_condition,
        use_furnace_output_buffer=task.use_furnace_output_buffer,
        use_furnace_input_buffer=task.use_furnace_input_buffer,
        use_miner_output_buffer=task.use_miner_output_buffer,
    )


def _run_episode(
    agent_name: str,
    task: TaskDefinition,
    seed: int,
    ppo_model: object | None,
) -> BenchmarkEpisode:
    env = _make_mock_env(task)

    if agent_name == "ppo":
        return _run_ppo_episode(env, task, seed, ppo_model)

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


def _run_ppo_episode(
    env: MockFactorioEnv,
    task: TaskDefinition,
    seed: int,
    ppo_model: object | None,
) -> BenchmarkEpisode:
    if ppo_model is None:
        raise ValueError("ppo_model is required when agent_name is 'ppo'.")

    wrapped_env = NumericObservationWrapper(env)
    observation, _ = wrapped_env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    invalid_actions = 0

    while not terminated and not truncated:
        action, _ = _predict_ppo_action(ppo_model, observation, wrapped_env)
        observation, reward, terminated, truncated, info = wrapped_env.step(int(action))
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
    raise ValueError("agent_name must be 'scripted', 'random', or 'ppo'.")


def _predict_ppo_action(
    model: object,
    observation: object,
    env: NumericObservationWrapper,
) -> tuple[object, object]:
    if model.__class__.__name__ == "MaskablePPO":
        return model.predict(  # type: ignore[attr-defined]
            observation,
            deterministic=True,
            action_masks=env.action_masks(),
        )
    return model.predict(observation, deterministic=True)  # type: ignore[attr-defined]
