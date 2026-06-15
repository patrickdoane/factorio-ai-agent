from factorio_ai_agent.cli import (
    build_parser,
    run_evaluate,
    run_random,
    run_research_benchmark,
    run_scripted,
)


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
    assert "Production: miner_progress=0 furnace_progress=0 target_iron_plates=1" in output


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
