import sys

import pytest

from factorio_ai_agent.training import train_ppo as train_ppo_module


def test_train_ppo_without_optional_dependency_prints_install_hint(
    capsys, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(train_ppo_module, "_runtime_supports_torch", lambda: True)
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)

    train_ppo_module.train_ppo(
        total_timesteps=1,
        task_name="first-plate",
        n_steps=64,
        batch_size=32,
    )

    output = capsys.readouterr().out

    assert "Stable-Baselines3 is not installed" in output


def test_train_ppo_preflight_prints_runtime_hint(capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(train_ppo_module, "_runtime_supports_torch", lambda: False)

    train_ppo_module.train_ppo(total_timesteps=1, task_name="first-plate")

    output = capsys.readouterr().out

    assert "PPO training requires a stable Python 3.11+ runtime" in output


def test_train_ppo_rejects_invalid_rollout_config() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        train_ppo_module.train_ppo(n_steps=32, batch_size=64)


def test_train_ppo_rejects_negative_eval_episodes() -> None:
    with pytest.raises(ValueError, match="eval_episodes"):
        train_ppo_module.train_ppo(eval_episodes=-1)
