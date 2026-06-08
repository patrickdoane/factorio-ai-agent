"""Command line interface for Factorio AI agent demos."""

from __future__ import annotations

import argparse

from factorio_ai_agent.agents.random_agent import RandomAgent
from factorio_ai_agent.agents.scripted_burner_agent import ScriptedBurnerAgent
from factorio_ai_agent.envs.mock_factorio_env import MockFactorioEnv
from factorio_ai_agent.training.train_ppo import train_ppo


def run_random(max_steps: int = 100) -> None:
    """Run the random agent in the mock environment."""
    env = MockFactorioEnv(max_steps=max_steps)
    agent = RandomAgent()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(env)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"{env.render()} reward={reward:.2f} info={info}")

    print(f"Finished random run: terminated={terminated} truncated={truncated}")


def run_scripted(max_steps: int = 100) -> None:
    """Run the scripted burner-miner agent in the mock environment."""
    env = MockFactorioEnv(max_steps=max_steps)
    agent = ScriptedBurnerAgent()
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"{env.render()} reward={reward:.2f} info={info}")

    print(f"Finished scripted run: terminated={terminated} truncated={truncated}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="factorio-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run-random", help="Run a random valid-action baseline.")
    subparsers.add_parser("run-scripted", help="Run the scripted burner-miner agent.")
    subparsers.add_parser("train-ppo", help="Run the optional PPO training entry point.")
    return parser


def main() -> None:
    """Run the requested CLI command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-random":
        run_random()
    elif args.command == "run-scripted":
        run_scripted()
    elif args.command == "train-ppo":
        train_ppo()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
