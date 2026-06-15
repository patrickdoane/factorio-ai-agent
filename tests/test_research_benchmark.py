import pytest

from factorio_ai_agent.research.benchmark import (
    append_results_tsv,
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


def test_append_results_tsv_writes_header_and_row(tmp_path) -> None:  # type: ignore[no-untyped-def]
    summary = run_benchmark(
        agent_name="scripted",
        task_names=["first-plate"],
        eval_episodes=1,
        seed=1,
    )
    result_path = tmp_path / "results.tsv"

    append_results_tsv(
        result_path,
        summary=summary,
        agent_name="scripted",
        task_names=["first-plate"],
        seed=1,
    )

    lines = result_path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == (
        "timestamp\tgit_commit\tagent\ttasks\tseed\tscore\tsuccess_rate\tavg_steps\t"
        "avg_reward\tinvalid_rate\teval_episodes"
    )
    assert len(lines) == 2
    row = lines[1].split("\t")
    assert row[2:] == [
        "scripted",
        "first-plate",
        "1",
        f"{summary.score:.6f}",
        f"{summary.success_rate:.6f}",
        f"{summary.avg_steps:.6f}",
        f"{summary.avg_reward:.6f}",
        f"{summary.invalid_rate:.6f}",
        "1",
    ]


def test_append_results_tsv_appends_without_repeating_header(tmp_path) -> None:  # type: ignore[no-untyped-def]
    summary = run_benchmark(
        agent_name="scripted",
        task_names=["first-plate"],
        eval_episodes=1,
        seed=1,
    )
    result_path = tmp_path / "results.tsv"

    for _ in range(2):
        append_results_tsv(
            result_path,
            summary=summary,
            agent_name="scripted",
            task_names=["first-plate"],
            seed=1,
        )

    lines = result_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert sum(line.startswith("timestamp\t") for line in lines) == 1


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
