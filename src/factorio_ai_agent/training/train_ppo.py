"""Optional Stable-Baselines3 PPO training entry point."""

from __future__ import annotations

import sys

from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv
from factorio_ai_agent.envs.numeric_observation_wrapper import NumericObservationWrapper


def train_ppo(total_timesteps: int = 1_000, device: str = "cpu") -> None:
    """Train PPO on the mock environment when optional RL dependencies exist."""
    if not _runtime_supports_torch():
        print(_runtime_error_message())
        return

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Stable-Baselines3 is not installed. Install with: pip install -e .[rl]")
        return

    env = NumericObservationWrapper(MockFactorioEnv())
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=total_timesteps)
    print("Finished PPO training demo.")


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
