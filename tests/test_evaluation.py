from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.evaluation import format_summary, run_episode, summarize_results


def test_run_episode_returns_successful_scripted_summary(capsys) -> None:  # type: ignore[no-untyped-def]
    agent = ScriptedBurnerAgent()

    result = run_episode(
        lambda _env, observation: agent.act(observation),
        max_steps=50,
        quiet=True,
    )

    assert result.success
    assert result.terminated
    assert not result.truncated
    assert result.steps == 20
    assert result.goal == "Produce 1 iron plate"
    assert result.status == "Task complete"
    assert result.final_objective == "Task complete"

    output = capsys.readouterr().out
    assert "=== Episode 1 Result ===" in output
    assert "Success: True" in output
    assert "Goal: Produce 1 iron plate" in output
    assert "Status: Task complete" in output


def test_summarize_results_handles_empty_and_non_empty_results() -> None:
    empty_summary = summarize_results([])

    assert empty_summary == {
        "episodes": 0.0,
        "success_rate": 0.0,
        "avg_steps": 0.0,
        "avg_reward": 0.0,
    }

    agent = ScriptedBurnerAgent()
    result = run_episode(
        lambda _env, observation: agent.act(observation),
        max_steps=50,
        quiet=True,
    )

    summary = summarize_results([result])

    assert summary["episodes"] == 1.0
    assert summary["success_rate"] == 1.0
    assert summary["avg_steps"] == 20.0


def test_format_summary_returns_human_readable_block() -> None:
    summary = {
        "episodes": 2.0,
        "success_rate": 0.5,
        "avg_steps": 10.0,
        "avg_reward": 1.25,
    }

    output = format_summary("Random", summary)

    assert "=== Random Summary ===" in output
    assert "Episodes: 2" in output
    assert "Success rate: 50.0%" in output
