from factorio_ai_agent.cli import run_evaluate, run_random, run_scripted


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


def test_evaluate_cli_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    run_evaluate(agent_name="both", episodes=1, max_steps=50, seed=3, verbose=False)

    output = capsys.readouterr().out

    assert "=== Scripted Summary ===" in output
    assert "=== Random Summary ===" in output
    assert "Episodes: 1" in output
