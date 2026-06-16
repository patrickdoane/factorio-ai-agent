"""Optional Stable-Baselines3 PPO training entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym

from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv, Observation
from factorio_ai_agent.envs.numeric_observation_wrapper import NumericObservationWrapper
from factorio_ai_agent.tasks import TaskDefinition, resolve_task
from factorio_ai_agent.training.reward_wrappers import (
    BurnerProgressRewardWrapper,
    ProgressRewardWrapper,
)


def train_ppo(
    total_timesteps: int = 1_000,
    device: str = "cpu",
    task_name: str = "first-plate",
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int | None = None,
    save_path: str | None = None,
    eval_episodes: int = 0,
    reward_shaping: str = "none",
    algo: str = "ppo",
    task_names: str | None = None,
) -> None:
    """Train PPO on the mock environment when optional RL dependencies exist."""
    _validate_ppo_config(
        n_steps=n_steps,
        batch_size=batch_size,
        eval_episodes=eval_episodes,
        reward_shaping=reward_shaping,
        algo=algo,
    )
    training_task_names = _parse_training_task_names(task_name, task_names)

    if not _runtime_supports_torch():
        print(_runtime_error_message())
        return

    try:
        model_class = _import_model_class(algo)
    except ImportError as error:
        print(f"{error} Install with: pip install -e '.[rl]'")
        return

    env = _make_training_envs(training_task_names, reward_shaping=reward_shaping)
    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        print(f"Saved {algo} model to {path}")

    if eval_episodes > 0:
        _evaluate_model(
            model,
            task_names=training_task_names,
            episodes=eval_episodes,
            seed=seed,
            label=algo,
        )

    print(f"Finished {algo} training demo.")


class MultiTaskTrainingEnv(gym.Env[Observation, int]):
    """Sample one configured mock task on each reset for PPO training."""

    def __init__(
        self,
        task_names: Sequence[str],
        *,
        reward_shaping: str = "none",
    ) -> None:
        super().__init__()
        self.tasks = [resolve_task(task_name) for task_name in task_names]
        self.reward_shaping = reward_shaping
        prototype = _make_mock_env(self.tasks[0], reward_shaping=reward_shaping)
        self.action_space = prototype.action_space
        self.observation_space = prototype.observation_space
        self.current_task_name = self.tasks[0].name
        self._current_env = prototype

    @property
    def step_count(self) -> int:
        return self._current_env.unwrapped.step_count  # type: ignore[attr-defined]

    def action_masks(self):  # type: ignore[no-untyped-def]
        return self._current_env.action_masks()  # type: ignore[attr-defined]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        task = self.tasks[int(self.np_random.integers(len(self.tasks)))]
        self.current_task_name = task.name
        self._current_env = _make_mock_env(task, reward_shaping=self.reward_shaping)
        observation, info = self._current_env.reset(seed=seed, options=options)
        info["task_name"] = task.name
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._current_env.step(action)
        info["task_name"] = self.current_task_name
        return observation, reward, terminated, truncated, info


def _import_model_class(algo: str) -> type:
    if algo == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise ImportError("Stable-Baselines3 is not installed.") from error
        return PPO
    if algo == "maskable-ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as error:
            raise ImportError("sb3-contrib is not installed.") from error
        return MaskablePPO
    raise ValueError("algo must be 'ppo' or 'maskable-ppo'.")


def _make_training_env(
    task_name: str,
    *,
    reward_shaping: str = "none",
) -> NumericObservationWrapper:
    task = resolve_task(task_name)
    return NumericObservationWrapper(_make_mock_env(task, reward_shaping=reward_shaping))


def _make_training_envs(
    task_names: Sequence[str],
    *,
    reward_shaping: str = "none",
) -> NumericObservationWrapper:
    if len(task_names) == 1:
        return _make_training_env(task_names[0], reward_shaping=reward_shaping)
    return NumericObservationWrapper(
        MultiTaskTrainingEnv(task_names, reward_shaping=reward_shaping)
    )


def _make_mock_env(
    task: TaskDefinition,
    *,
    reward_shaping: str = "none",
) -> gym.Env[Observation, int]:
    env = MockFactorioEnv(
        max_steps=task.max_steps,
        target_iron_plates=task.target_iron_plates,
        require_burner_miner_for_success=task.require_burner_miner_for_success,
        starting_inventory=dict(task.starting_inventory),
        required_burner_mined_iron_ore=task.required_burner_mined_iron_ore,
    )
    if reward_shaping == "progress":
        env = ProgressRewardWrapper(env)  # type: ignore[assignment]
    if reward_shaping == "burner-progress":
        env = BurnerProgressRewardWrapper(env)  # type: ignore[assignment]
    return env


def _parse_training_task_names(task_name: str, task_names: str | None) -> list[str]:
    if task_names is None:
        resolve_task(task_name)
        return [task_name]
    names = [name.strip() for name in task_names.split(",") if name.strip()]
    if not names:
        raise ValueError("task_names must include at least one task.")
    for name in names:
        resolve_task(name)
    return names


def _evaluate_model(
    model: object,
    *,
    task_names: Sequence[str],
    episodes: int,
    seed: int | None,
    label: str = "ppo",
) -> None:
    successes = 0
    total_steps = 0
    total_episodes = episodes * len(task_names)

    for task_index, task_name in enumerate(task_names):
        for episode in range(episodes):
            env = _make_training_env(task_name)
            episode_seed = None
            if seed is not None:
                episode_seed = seed + task_index * episodes + episode
            observation, _ = env.reset(seed=episode_seed)
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action, _ = _predict_action(model, observation, env, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(int(action))

            successes += int(terminated)
            total_steps += env.unwrapped.step_count

    success_rate = successes / total_episodes
    avg_steps = total_steps / total_episodes
    print(
        f"{label} eval: "
        f"tasks={','.join(task_names)} episodes={total_episodes} "
        f"success_rate={success_rate * 100:.1f}% "
        f"avg_steps={avg_steps:.1f}"
    )


def _validate_ppo_config(
    n_steps: int,
    batch_size: int,
    eval_episodes: int,
    reward_shaping: str = "none",
    algo: str = "ppo",
) -> None:
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2 for PPO.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if batch_size > n_steps:
        raise ValueError("batch_size must be less than or equal to n_steps.")
    if eval_episodes < 0:
        raise ValueError("eval_episodes must be zero or greater.")
    if reward_shaping not in {"none", "progress", "burner-progress"}:
        raise ValueError(
            "reward_shaping must be 'none', 'progress', or 'burner-progress'."
        )
    if algo not in {"ppo", "maskable-ppo"}:
        raise ValueError("algo must be 'ppo' or 'maskable-ppo'.")


def _predict_action(
    model: object,
    observation: object,
    env: NumericObservationWrapper,
    *,
    deterministic: bool,
) -> tuple[object, object]:
    if model.__class__.__name__ == "MaskablePPO":
        return model.predict(  # type: ignore[attr-defined]
            observation,
            deterministic=deterministic,
            action_masks=env.action_masks(),
        )
    return model.predict(observation, deterministic=deterministic)  # type: ignore[attr-defined]


def _runtime_supports_torch() -> bool:
    """Return whether the current Python runtime is suitable for Torch/SB3."""
    return (
        sys.version_info >= (3, 11)
        and sys.version_info.releaselevel == "final"
        and hasattr(sys, "get_int_max_str_digits")
    )


def _runtime_error_message() -> str:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info.releaselevel != "final":
        version = f"{version}{sys.version_info.releaselevel}{sys.version_info.serial}"
    return (
        "PPO training requires a stable Python 3.11+ runtime compatible with "
        f"Torch. Current runtime is Python {version}. Reinstall a stable "
        "Python 3.11 release, recreate .venv, then run: pip install -e '.[dev,rl]'"
    )
