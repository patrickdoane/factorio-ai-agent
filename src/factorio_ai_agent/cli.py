"""Command line interface for Factorio AI agent demos."""

from __future__ import annotations

import argparse

from factorio_ai_agent.agents.random_agent import RandomAgent
from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.evaluation import format_summary, run_episode, summarize_results
from factorio_ai_agent.training.train_ppo import train_ppo


def run_random(max_steps: int = 100, episodes: int = 1, quiet: bool = False) -> None:
    """Run the random agent in the mock environment."""
    agent = RandomAgent()
    results = []

    for episode_number in range(1, episodes + 1):
        results.append(
            run_episode(
                lambda env, _observation: agent.act(env),
                max_steps=max_steps,
                quiet=quiet,
                agent_name="random",
                episode_number=episode_number,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="factorio-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser(
        "run-random", help="Run a random valid-action baseline."
    )
    random_parser.add_argument("--max-steps", type=int, default=100)
    random_parser.add_argument("--episodes", type=int, default=1)
    random_parser.add_argument("--quiet", action="store_true")

    scripted_parser = subparsers.add_parser(
        "run-scripted", help="Run the scripted burner-miner agent."
    )
    scripted_parser.add_argument("--max-steps", type=int, default=100)
    scripted_parser.add_argument("--quiet", action="store_true")

    subparsers.add_parser("train-ppo", help="Run the optional PPO training entry point.")
    return parser


def main() -> None:
    """Run the requested CLI command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-random":
        run_random(max_steps=args.max_steps, episodes=args.episodes, quiet=args.quiet)
    elif args.command == "run-scripted":
        run_scripted(max_steps=args.max_steps, quiet=args.quiet)
    elif args.command == "train-ppo":
        train_ppo()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
