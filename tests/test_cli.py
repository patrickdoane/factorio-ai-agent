import pytest

import factorio_ai_agent.cli as cli_module
from factorio_ai_agent.envs.mock_factorio_env import Action
from factorio_ai_agent.cli import (
    build_parser,
    run_evaluate,
    run_ppo,
    run_random,
    run_research_benchmark,
    run_scripted,
)


class FakePPOModel:
    def predict(self, observation, deterministic: bool = True):  # type: ignore[no-untyped-def]
        return int(Action.WAIT), None


def test_run_scripted_cli_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    run_scripted(max_steps=50, quiet=True)

    output = capsys.readouterr().out

    assert "=== Episode 1 Result ===" in output
    assert "Success: True" in output
    assert "Truncated: False" in output


def test_run_random_multiple_episodes_cli_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    run_random(max_steps=1, episodes=2, quiet=True, seed=7)

    output = capsys.readouterr().out

    assert "=== Random Summary ===" in output
    assert "Episodes: 2" in output


def test_run_scripted_verbose_output_has_readable_step_blocks(capsys) -> None:  # type: ignore[no-untyped-def]
    run_scripted(max_steps=50, quiet=False)

    output = capsys.readouterr().out

    assert "=== Episode 1: scripted ===" in output
    assert "Step 01" in output
    assert "Action: MINE_STONE" in output
    assert "Inventory: iron_ore=0 coal=0 stone=1" in output
    assert (
        "Production: miner_progress=0 furnace_progress=0 target_iron_plates=1 "
        "burner_mined_iron_ore=0"
    ) in output


def test_run_scripted_supports_target_iron_plates(capsys) -> None:  # type: ignore[no-untyped-def]
    run_scripted(quiet=True, task_name="three-plates")

    output = capsys.readouterr().out

    assert "Success: True" in output
    assert "Goal: Produce 3 iron plates" in output
    assert "Status: Task complete" in output


def test_evaluate_cli_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    run_evaluate(
        agent_name="both",
        episodes=1,
        seed=3,
        verbose=False,
        task_name="three-plates",
    )

    output = capsys.readouterr().out

    assert "=== Scripted Summary ===" in output
    assert "=== Random Summary ===" in output
    assert "Episodes: 1" in output


def test_train_ppo_parser_accepts_quick_run_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train-ppo",
            "--task",
            "first-plate",
            "--total-timesteps",
            "64",
            "--n-steps",
            "64",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.001",
            "--seed",
            "42",
            "--save-path",
            "models/test.zip",
            "--eval-episodes",
            "2",
            "--reward-shaping",
            "burner-progress",
        ]
    )

    assert args.command == "train-ppo"
    assert args.total_timesteps == 64
    assert args.n_steps == 64
    assert args.batch_size == 32
    assert args.learning_rate == 0.001
    assert args.seed == 42
    assert args.save_path == "models/test.zip"
    assert args.eval_episodes == 2
    assert args.reward_shaping == "burner-progress"


def test_run_ppo_parser_accepts_rollout_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run-ppo",
            "--model-path",
            "models/test.zip",
            "--task",
            "first-plate",
            "--seed",
            "42",
            "--max-steps",
            "3",
        ]
    )

    assert args.command == "run-ppo"
    assert args.model_path == "models/test.zip"
    assert args.task == "first-plate"
    assert args.seed == 42
    assert args.max_steps == 3


def test_run_ppo_prints_readable_rollout(capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli_module, "_load_ppo_model", lambda _path: FakePPOModel())

    run_ppo(model_path="models/test.zip", max_steps=2, seed=42)

    output = capsys.readouterr().out

    assert "=== Episode 1: ppo ===" in output
    assert "Step 01" in output
    assert "Action: WAIT" in output
    assert "Inventory: iron_ore=0 coal=0 stone=0" in output
    assert "=== Episode 1 Result ===" in output
    assert "Success: False" in output
    assert "Truncated: True" in output


def test_research_benchmark_cli_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    run_research_benchmark(
        agent_name="scripted",
        tasks="first-plate,three-plates",
        eval_episodes=1,
        seed=42,
    )

    output = capsys.readouterr().out

    assert output.startswith("---\n")
    assert "score:" in output
    assert "success_rate:       1.000000" in output


def test_research_benchmark_parser_accepts_append_results() -> None:
    parser = build_parser()

    args = parser.parse_args(["research-benchmark", "--append-results"])

    assert args.command == "research-benchmark"
    assert args.append_results == "results.tsv"


def test_research_benchmark_parser_accepts_ppo_model_path() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["research-benchmark", "--agent", "ppo", "--model-path", "models/test.zip"]
    )

    assert args.command == "research-benchmark"
    assert args.agent == "ppo"
    assert args.model_path == "models/test.zip"


def test_research_benchmark_cli_requires_ppo_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        run_research_benchmark(
            agent_name="ppo",
            tasks="first-plate",
            eval_episodes=1,
            seed=42,
        )


def test_research_benchmark_cli_appends_results(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    result_path = tmp_path / "bench.tsv"

    run_research_benchmark(
        agent_name="scripted",
        tasks="first-plate",
        eval_episodes=1,
        seed=42,
        append_results=str(result_path),
    )

    output = capsys.readouterr().out

    assert "Appended benchmark result to" in output
    assert "scripted\tfirst-plate\t42" in result_path.read_text(encoding="utf-8")
