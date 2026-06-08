"""Command line interface for Factorio AI agent demos."""

from __future__ import annotations

import argparse

from factorio_ai_agent.agents.random_agent import RandomAgent
from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.evaluation import format_summary, run_episode, summarize_results
from factorio_ai_agent.training.train_ppo import train_ppo


def run_random(
    max_steps: int = 100,
    episodes: int = 1,
    quiet: bool = False,
    seed: int | None = None,
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
            )
        )

    if episodes > 1:
        print(format_summary("Random", summarize_results(results)))


def run_scripted(max_steps: int = 100, quiet: bool = False) -> None:
    """Run the scripted burner-miner agent in the mock environment."""
    agent = ScriptedBurnerAgent()
    run_episode(
        lambda _env, observation: agent.act(observation),
        max_steps=max_steps,
        quiet=quiet,
        agent_name="scripted",
        episode_number=1,
    )


def run_evaluate(
    agent_name: str = "both",
    episodes: int = 10,
    max_steps: int = 100,
    seed: int | None = None,
    verbose: bool = False,
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
            )
            for episode_number in range(1, episodes + 1)
        ]
        print(format_summary(selected_agent.title(), summarize_results(results)))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="factorio-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser(
        "run-random", help="Run a random valid-action baseline."
    )
    random_parser.add_argument("--max-steps", type=int, default=100)
    random_parser.add_argument("--episodes", type=int, default=1)
    random_parser.add_argument("--seed", type=int, default=None)
    random_parser.add_argument("--quiet", action="store_true")

    scripted_parser = subparsers.add_parser(
        "run-scripted", help="Run the scripted burner-miner agent."
    )
    scripted_parser.add_argument("--max-steps", type=int, default=100)
    scripted_parser.add_argument("--quiet", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare baseline agents over multiple episodes."
    )
    evaluate_parser.add_argument(
        "--agent", choices=["scripted", "random", "both"], default="both"
    )
    evaluate_parser.add_argument("--episodes", type=int, default=10)
    evaluate_parser.add_argument("--max-steps", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=None)
    evaluate_parser.add_argument("--verbose", action="store_true")

    train_parser = subparsers.add_parser(
        "train-ppo", help="Run the optional PPO training entry point."
    )
    train_parser.add_argument("--total-timesteps", type=int, default=1_000)
    train_parser.add_argument("--device", default="cpu")
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
        )
    elif args.command == "run-scripted":
        run_scripted(max_steps=args.max_steps, quiet=args.quiet)
    elif args.command == "evaluate":
        run_evaluate(
            agent_name=args.agent,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=args.verbose,
        )
    elif args.command == "train-ppo":
        train_ppo(total_timesteps=args.total_timesteps, device=args.device)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
