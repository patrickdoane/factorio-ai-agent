"""Optional Stable-Baselines3 PPO training entry point."""

from __future__ import annotations

from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv


def train_ppo(total_timesteps: int = 1_000) -> None:
    """Train PPO on the mock environment when optional RL dependencies exist."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Stable-Baselines3 is not installed. Install with: pip install -e .[rl]")
        return

    env = MockFactorioEnv()
    # TODO: Add an RL-specific observation wrapper before serious PPO training.
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    print("Finished PPO training demo.")
