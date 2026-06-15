"""Command line interface for Factorio AI agent demos."""

from __future__ import annotations

import argparse

from factorio_ai_agent.agents.random_agent import RandomAgent
from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.evaluation import format_summary, run_episode, summarize_results
from factorio_ai_agent.research.benchmark import (
    append_results_tsv,
    format_benchmark_summary,
    parse_task_names,
    run_benchmark,
)
from factorio_ai_agent.tasks import resolve_task, task_names
from factorio_ai_agent.training.train_ppo import train_ppo


def run_random(
    max_steps: int | None = None,
    episodes: int = 1,
    quiet: bool = False,
    seed: int | None = None,
    target_iron_plates: int | None = None,
    task_name: str = "first-plate",
) -> None:
    """Run the random agent in the mock environment."""
    agent = RandomAgent(seed=seed)
    results = []

    for episode_number in range(1, episodes + 1):
        results.append(
            run_episode(
                lambda env, _observation: agent.act(env),
                max_steps=max_steps,
                quiet=quiet,
                agent_name="random",
                episode_number=episode_number,
                seed=seed,
                target_iron_plates=target_iron_plates,
                task_name=task_name,
            )
        )

    if episodes > 1:
        print(format_summary("Random", summarize_results(results)))


def run_scripted(
    max_steps: int | None = None,
    quiet: bool = False,
    target_iron_plates: int | None = None,
    task_name: str = "first-plate",
) -> None:
    """Run the scripted burner-miner agent in the mock environment."""
    agent = ScriptedBurnerAgent()
    run_episode(
        lambda _env, observation: agent.act(observation),
        max_steps=max_steps,
        quiet=quiet,
        agent_name="scripted",
        episode_number=1,
        target_iron_plates=target_iron_plates,
        task_name=task_name,
    )


def run_evaluate(
    agent_name: str = "both",
    episodes: int = 10,
    max_steps: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
    target_iron_plates: int | None = None,
    task_name: str = "first-plate",
) -> None:
    """Evaluate one or both baseline agents over multiple episodes."""
    selected_agents = ["scripted", "random"] if agent_name == "both" else [agent_name]

    for selected_agent in selected_agents:
        if selected_agent == "scripted":
            agent = ScriptedBurnerAgent()
            selector = lambda _env, observation: agent.act(observation)
        else:
            agent = RandomAgent(seed=seed)
            selector = lambda env, _observation: agent.act(env)

        results = [
            run_episode(
                selector,
                max_steps=max_steps,
                quiet=not verbose,
                agent_name=selected_agent,
                episode_number=episode_number,
                seed=seed,
                target_iron_plates=target_iron_plates,
                task_name=task_name,
            )
            for episode_number in range(1, episodes + 1)
        ]
        print(format_summary(selected_agent.title(), summarize_results(results)))


def run_research_benchmark(
    agent_name: str,
    tasks: str,
    eval_episodes: int,
    seed: int,
    append_results: str | None = None,
) -> None:
    """Run the deterministic research benchmark and print the final summary."""
    task_names = parse_task_names(tasks)
    summary = run_benchmark(
        agent_name=agent_name,
        task_names=task_names,
        eval_episodes=eval_episodes,
        seed=seed,
    )
    print(format_benchmark_summary(summary))
    if append_results:
        path = append_results_tsv(
            append_results,
            summary=summary,
            agent_name=agent_name,
            task_names=task_names,
            seed=seed,
        )
        print(f"Appended benchmark result to {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="factorio-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser(
        "run-random", help="Run a random valid-action baseline."
    )
    random_parser.add_argument("--task", choices=task_names(), default="first-plate")
    random_parser.add_argument("--max-steps", type=int, default=None)
    random_parser.add_argument("--episodes", type=int, default=1)
    random_parser.add_argument("--seed", type=int, default=None)
    random_parser.add_argument("--target-iron-plates", type=int, default=None)
    random_parser.add_argument("--quiet", action="store_true")

    scripted_parser = subparsers.add_parser(
        "run-scripted", help="Run the scripted burner-miner agent."
    )
    scripted_parser.add_argument("--task", choices=task_names(), default="first-plate")
    scripted_parser.add_argument("--max-steps", type=int, default=None)
    scripted_parser.add_argument("--target-iron-plates", type=int, default=None)
    scripted_parser.add_argument("--quiet", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare baseline agents over multiple episodes."
    )
    evaluate_parser.add_argument(
        "--agent", choices=["scripted", "random", "both"], default="both"
    )
    evaluate_parser.add_argument("--task", choices=task_names(), default="first-plate")
    evaluate_parser.add_argument("--episodes", type=int, default=10)
    evaluate_parser.add_argument("--max-steps", type=int, default=None)
    evaluate_parser.add_argument("--seed", type=int, default=None)
    evaluate_parser.add_argument("--target-iron-plates", type=int, default=None)
    evaluate_parser.add_argument("--verbose", action="store_true")

    benchmark_parser = subparsers.add_parser(
        "research-benchmark",
        help="Run a deterministic benchmark for autonomous research loops.",
    )
    benchmark_parser.add_argument("--agent", choices=["scripted", "random"], default="scripted")
    benchmark_parser.add_argument("--tasks", default="first-plate,three-plates")
    benchmark_parser.add_argument("--eval-episodes", type=int, default=10)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument(
        "--append-results",
        nargs="?",
        const="results.tsv",
        default=None,
        help="Append the benchmark summary to a TSV file. Defaults to results.tsv.",
    )

    train_parser = subparsers.add_parser(
        "train-ppo", help="Run the optional PPO training entry point."
    )
    train_parser.add_argument("--task", choices=task_names(), default="first-plate")
    train_parser.add_argument("--total-timesteps", type=int, default=1_000)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--n-steps", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--save-path", default=None)
    train_parser.add_argument("--eval-episodes", type=int, default=0)

    tasks_parser = subparsers.add_parser("list-tasks", help="List available mock tasks.")
    tasks_parser.set_defaults(command="list-tasks")
    return parser


def main() -> None:
    """Run the requested CLI command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-random":
        run_random(
            max_steps=args.max_steps,
            episodes=args.episodes,
            quiet=args.quiet,
            seed=args.seed,
            target_iron_plates=args.target_iron_plates,
            task_name=args.task,
        )
    elif args.command == "run-scripted":
        run_scripted(
            max_steps=args.max_steps,
            quiet=args.quiet,
            target_iron_plates=args.target_iron_plates,
            task_name=args.task,
        )
    elif args.command == "evaluate":
        run_evaluate(
            agent_name=args.agent,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=args.verbose,
            target_iron_plates=args.target_iron_plates,
            task_name=args.task,
        )
    elif args.command == "research-benchmark":
        run_research_benchmark(
            agent_name=args.agent,
            tasks=args.tasks,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            append_results=args.append_results,
        )
    elif args.command == "train-ppo":
        train_ppo(
            total_timesteps=args.total_timesteps,
            device=args.device,
            task_name=args.task,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            save_path=args.save_path,
            eval_episodes=args.eval_episodes,
        )
    elif args.command == "list-tasks":
        for task_name in task_names():
            task = resolve_task(task_name)
            print(
                f"{task.name}: {task.description} "
                f"target_iron_plates={task.target_iron_plates} max_steps={task.max_steps}"
            )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
