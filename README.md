# factorio-ai-agent

A prototype AI agent framework for small Factorio-like tasks.

The project starts with a lightweight mock environment so agents can be built,
tested, and trained locally before any real Factorio headless server integration
exists. The first task is intentionally tiny: mine resources, craft and place a
stone furnace and burner mining drill, fuel the setup, and produce the first iron
plate.

## Goals

- Provide a Gymnasium-style environment interface.
- Keep the first environment deterministic and easy to test.
- Support simple baseline agents before adding reinforcement learning.
- Leave clear adapter seams for future Factorio headless server integration.
- Optionally support Stable-Baselines3 PPO training later.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Or use the local development shortcut:

```bash
make setup
```

To install optional reinforcement learning dependencies:

```bash
pip install -e .[dev,rl]
```

## Usage

Run a random agent:

```bash
factorio-ai run-random --episodes 3 --max-steps 100 --quiet
```

Run the scripted burner-miner agent:

```bash
factorio-ai run-scripted --max-steps 50
```

Run the PPO training entry point:

```bash
factorio-ai train-ppo
```

## Mock Task

`MockFactorioEnv` simulates a minimal Factorio task with discrete actions:

- Mine iron ore, coal, and stone.
- Craft a stone furnace.
- Craft a burner mining drill.
- Place the furnace and burner miner.
- Insert coal fuel.
- Wait for the first iron plate to be produced.

The observation is a dictionary containing:

- `inventory`
- `placed_entities`
- `step_count`
- `current_objective`

## Roadmap

1. Expand the mock environment with positions, entity orientation, and recipes.
2. Add a real Factorio adapter using a headless server plus RCON or a mod bridge.
3. Convert observations/actions into RL-friendly spaces for Stable-Baselines3.
4. Add curriculum tasks for mining, crafting, placement, and automation.
5. Benchmark scripted, random, and trained agents across fixed scenarios.

## Development

```bash
source .venv/bin/activate
pytest
```

With `make`:

```bash
make test
make run-scripted
```

The CLI supports `--quiet` for summary-only runs. `run-random` also supports
`--episodes` for quick baseline evaluation.

This repository is intended to use a lightweight feature branch workflow for
rapid local iteration.

Recommended workflow:

```bash
git checkout -b feature/small-change
make test
git status
```

Keep each branch focused on one small change so the mock environment, agents,
and future Factorio adapter can evolve independently.
