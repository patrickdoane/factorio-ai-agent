import pytest

from factorio_ai_agent.research.benchmark import (
    format_benchmark_summary,
    parse_task_names,
    run_benchmark,
)


def test_scripted_research_benchmark_is_deterministic() -> None:
    first = run_benchmark(
        agent_name="scripted",
        task_names=["first-plate", "three-plates"],
        eval_episodes=2,
        seed=42,
    )
    second = run_benchmark(
        agent_name="scripted",
        task_names=["first-plate", "three-plates"],
        eval_episodes=2,
        seed=42,
    )

    assert first == second
    assert first.success_rate == 1.0
    assert first.invalid_rate == 0.0
    assert first.eval_episodes == 4


def test_random_research_benchmark_returns_summary() -> None:
    summary = run_benchmark(
        agent_name="random",
        task_names=["first-plate"],
        eval_episodes=2,
        seed=7,
    )

    assert summary.score <= 1.0
    assert 0.0 <= summary.success_rate <= 1.0
    assert summary.avg_steps > 0.0
    assert summary.eval_episodes == 2


def test_research_benchmark_summary_format_is_machine_readable() -> None:
    summary = run_benchmark(
        agent_name="scripted",
        task_names=["first-plate"],
        eval_episodes=1,
        seed=1,
    )

    output = format_benchmark_summary(summary)

    assert output.startswith("---\n")
    assert "score:" in output
    assert "success_rate:" in output
    assert "avg_steps:" in output
    assert "avg_reward:" in output
    assert "invalid_rate:" in output
    assert "eval_episodes:      1" in output


def test_parse_task_names_rejects_unknown_tasks() -> None:
    assert parse_task_names("first-plate, three-plates") == [
        "first-plate",
        "three-plates",
    ]

    with pytest.raises(ValueError, match="Unknown task"):
        parse_task_names("unknown")


def test_research_benchmark_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="eval_episodes"):
        run_benchmark(
            agent_name="scripted",
            task_names=["first-plate"],
            eval_episodes=0,
            seed=1,
        )

    with pytest.raises(ValueError, match="agent_name"):
        run_benchmark(
            agent_name="ppo",
            task_names=["first-plate"],
            eval_episodes=1,
            seed=1,
        )
