"""Optional Stable-Baselines3 PPO training entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv
from factorio_ai_agent.envs.numeric_observation_wrapper import NumericObservationWrapper
from factorio_ai_agent.tasks import resolve_task
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
) -> None:
    """Train PPO on the mock environment when optional RL dependencies exist."""
    _validate_ppo_config(
        n_steps=n_steps,
        batch_size=batch_size,
        eval_episodes=eval_episodes,
        reward_shaping=reward_shaping,
        algo=algo,
    )

    if not _runtime_supports_torch():
        print(_runtime_error_message())
        return

    try:
        model_class = _import_model_class(algo)
    except ImportError as error:
        print(f"{error} Install with: pip install -e '.[rl]'")
        return

    env = _make_training_env(task_name, reward_shaping=reward_shaping)
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
            task_name=task_name,
            episodes=eval_episodes,
            seed=seed,
            label=algo,
        )

    print(f"Finished {algo} training demo.")


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
    return NumericObservationWrapper(
        env
    )


def _evaluate_model(
    model: object,
    *,
    task_name: str,
    episodes: int,
    seed: int | None,
    label: str = "ppo",
) -> None:
    successes = 0
    total_steps = 0

    for episode in range(episodes):
        env = _make_training_env(task_name)
        observation, _ = env.reset(seed=None if seed is None else seed + episode)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = _predict_action(model, observation, env, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(int(action))

        successes += int(terminated)
        total_steps += env.unwrapped.step_count

    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    print(
        f"{label} eval: "
        f"episodes={episodes} success_rate={success_rate * 100:.1f}% "
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
