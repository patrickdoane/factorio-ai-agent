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
factorio-ai run-random --episodes 3 --max-steps 100 --seed 42 --quiet
```

Run the scripted burner-miner agent:

```bash
factorio-ai run-scripted --max-steps 50
```

Run the PPO training entry point:

```bash
factorio-ai train-ppo --total-timesteps 2048
```

PPO defaults to `--device cpu` because this prototype uses a small MLP policy;
that avoids poor GPU utilization warnings, especially under WSL. You can still
override it with `--device cuda` when needed. The command also checks for a
stable Python 3.11+ runtime before importing Stable-Baselines3/Torch.

Compare baselines over multiple episodes:

```bash
factorio-ai evaluate --agent both --episodes 10 --max-steps 100 --seed 42
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

For reinforcement learning experiments, `NumericObservationWrapper` converts the
mock dictionary observation into a fixed numeric vector and drops the string
objective. This keeps the default environment readable while giving PPO a simple
`Box` observation space.

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
make evaluate
```

The CLI supports `--quiet` for summary-only runs. `run-random` also supports
`--episodes` for quick baseline evaluation. Use `factorio-ai evaluate` for a
summary-first comparison of scripted and random agents.

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
