from factorio_ai_agent.cli import run_scripted


def test_run_scripted_cli_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    run_scripted(max_steps=50)

    output = capsys.readouterr().out

    assert "Finished scripted run: terminated=True truncated=False" in output
