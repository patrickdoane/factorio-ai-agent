# factorio-ai-agent

A prototype AI agent framework for small Factorio-like tasks.

The project starts with a lightweight mock environment so agents can be built,
tested, and trained locally before any real Factorio headless server integration
exists. The first task is intentionally tiny: mine resources, craft and place a
stone furnace and burner mining drill, fuel the setup, and produce the first iron
plate. The mock environment can also target multiple iron plates using simple
wait-based miner and furnace production timing.

## Vision

The long-term goal is to build AI agents that can autonomously play Factorio,
progressing from reliable primitive production tasks to full rocket launch and
eventually speedrun-style optimization. The mock environment is a controlled,
deterministic training and evaluation ladder before real Factorio integration;
benchmark improvements should reflect better agent behavior, not changed task or
scoring semantics. See `docs/project-vision.md` for durable handoff context.
Current milestone results and reproduction commands are recorded in
`docs/milestones.md`.

## Goals

- Provide a Gymnasium-style environment interface.
- Keep the first environment deterministic and easy to test.
- Support simple baseline agents before adding reinforcement learning.
- Leave clear adapter seams for future Factorio headless server integration.
- Optionally support Stable-Baselines3 PPO training later.

## Install

Recommended reproducible setup with `uv`, including optional RL/PPO dependencies:

```bash
uv sync --all-extras
uv run pytest
uv run factorio-ai list-tasks
```

The checked-in `uv.lock` pins the full development and optional RL dependency
set so the project can be recreated consistently across machines.

For a lighter non-RL development environment, use:

```bash
uv sync --extra dev
```

Fallback setup with `venv` and `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Or use the local development shortcut:

```bash
make setup
```

`make setup` installs the lightweight development environment. To install
optional reinforcement learning dependencies for PPO training and benchmarking:

```bash
make setup-rl
```

Without `uv`, the Makefile falls back to `python -m venv` plus `pip install -e
'.[dev]'` or `pip install -e '.[dev,rl]'`.

## Usage

Run a random agent:

```bash
factorio-ai run-random --task first-plate --episodes 3 --seed 42 --quiet
```

Run the scripted burner-miner agent:

```bash
factorio-ai run-scripted --task three-plates
```

For explicit burner-chain scenarios, use `burner-first-plate`,
`burner-three-plates`, or `burner-ten-plates`.

For empty-inventory bootstrap curriculum, use `bootstrap-craft-furnace`,
`bootstrap-smelt-plates`, `bootstrap-craft-drill`, or
`bootstrap-place-and-fuel-drill`. These isolate the subskills needed before the
full `burner-first-plate` task: crafting a furnace, smelting drill ingredients,
crafting the drill, and placing/fueling it.

For machine-buffer scenarios, use `buffered-smelt-plate`,
`buffered-collect-plate`, or `buffered-collect-three-plates`. These tasks keep
smelted plates in the furnace output buffer instead of moving them directly into
inventory; the collection variants require explicitly taking the furnace output.
For stricter furnace I/O scenarios, use `buffered-insert-smelt-plate`,
`buffered-insert-collect-plate`, or `buffered-insert-collect-three-plates`.
These require explicitly inserting ore into the furnace input buffer before
smelting can advance.
For burner miner output scenarios, use `buffered-miner-output-ore` or
`buffered-miner-collect-ore`. These keep burner-mined ore in the miner output
buffer until the collection variant explicitly takes it.

For Freeplay-style crashland starts, use `freeplay-burner-first-plate`,
`freeplay-burner-three-plates`, or `freeplay-burner-ten-plates`. These tasks
start with `1` burner mining drill, `1` stone furnace, and `8` iron plates.

Run the PPO training entry point:

```bash
factorio-ai train-ppo --task first-plate --total-timesteps 256 --n-steps 64 --batch-size 32 --eval-episodes 3
```

Train with mask-aware PPO when valid-action masking matters:

```bash
factorio-ai train-ppo --algo maskable-ppo --task freeplay-burner-first-plate --total-timesteps 50000 --reward-shaping burner-progress --save-path /tmp/opencode/maskable-freeplay-first.zip
```

Train one policy across multiple tasks by passing a comma-separated task list:

```bash
factorio-ai train-ppo --algo maskable-ppo --tasks freeplay-burner-first-plate,freeplay-burner-three-plates,freeplay-burner-ten-plates --total-timesteps 50000 --reward-shaping burner-progress --save-path /tmp/opencode/maskable-freeplay-multitask.zip
```

Continue training from an existing saved policy with `--load-path`:

```bash
factorio-ai train-ppo --algo maskable-ppo --load-path /tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip --tasks freeplay-burner-first-plate,freeplay-burner-three-plates,freeplay-burner-ten-plates --total-timesteps 25000 --reward-shaping burner-progress --save-path /tmp/opencode/maskable-freeplay-multitask-finetuned.zip
```

Use training-only progress reward shaping for denser PPO feedback while keeping
benchmark rewards unchanged:

```bash
factorio-ai train-ppo --task first-plate --total-timesteps 50000 --reward-shaping progress --save-path /tmp/opencode/ppo-shaped-first.zip
```

For explicit burner-chain tasks, use `burner-progress` shaping so manual plate
shortcuts do not receive the large completion reward during training:

```bash
factorio-ai train-ppo --task burner-first-plate --total-timesteps 50000 --reward-shaping burner-progress --save-path /tmp/opencode/ppo-burner-first.zip
```

Inspect a saved PPO policy with legible step-by-step output:

```bash
factorio-ai run-ppo --model-path models/ppo-first.zip --task first-plate --seed 42
```

Use `--model-algo maskable-ppo` when inspecting or benchmarking a MaskablePPO
model:

```bash
factorio-ai run-ppo --model-path models/maskable-first.zip --model-algo maskable-ppo --task freeplay-burner-first-plate --seed 42
```

PPO and MaskablePPO require the optional `rl` dependency extra. Install it with
`uv sync --all-extras` or `make setup-rl`. PPO defaults to `--device cpu` because this
prototype uses a small MLP policy; that avoids poor GPU utilization warnings,
especially under WSL. You can still override it with `--device cuda` when needed.
The command also checks for a stable Python 3.11+ runtime before importing
Stable-Baselines3/Torch. PPO's rollout size is controlled by `--n-steps`; lower
values are useful for smoke tests, while larger values are better for real
training runs.

Compare baselines over multiple episodes:

```bash
factorio-ai evaluate --agent both --task three-plates --episodes 10 --seed 42
```

List available tasks:

```bash
factorio-ai list-tasks
```

Run the deterministic research benchmark:

```bash
factorio-ai research-benchmark --agent scripted --tasks first-plate,three-plates --eval-episodes 10 --seed 42
```

Benchmark a saved PPO model against the same fixed tasks:

```bash
factorio-ai research-benchmark --agent ppo --model-path models/ppo-first.zip --tasks first-plate,three-plates --eval-episodes 10 --seed 42
```

Append benchmark results to untracked `results.tsv`:

```bash
factorio-ai research-benchmark --agent scripted --tasks first-plate,three-plates --eval-episodes 10 --seed 42 --append-results
```

## Mock Task

`MockFactorioEnv` simulates a minimal Factorio task with discrete actions:

- Mine iron ore, coal, and stone.
- Craft a stone furnace.
- Craft iron gear wheels.
- Craft a burner mining drill.
- Place the furnace and burner miner.
- Insert coal fuel.
- Insert iron ore into a furnace input buffer.
- Take burner miner output.
- Take furnace output.
- Wait for the miner and furnace production timers to produce iron plates.

The mock early-game recipes are intentionally closer to vanilla Factorio than
the original prototype:

- `5 stone -> 1 stone_furnace`
- `2 iron_plate -> 1 iron_gear_wheel`
- `3 iron_plate + 3 iron_gear_wheel + 1 stone_furnace -> 1 burner_mining_drill`

The observation is a dictionary containing:

- `inventory`
- `placed_entities`
- `production_state`
- `step_count`
- `current_objective`

`production_state` tracks miner progress, furnace progress, burner-mined ore,
burner miner output buffers, furnace input/output buffers, and the target iron
plate count for the episode. This is still a simplified model: it captures the
need to wait for production, feed furnace input, and collect machine output
without modeling positions, belts, inserters, or furnace fuel separately yet.

Named tasks currently include the legacy simplified tasks `first-plate`,
`three-plates`, and `ten-plates`, plus explicit `manual-first-plate`,
`burner-first-plate`, `burner-three-plates`, and `burner-ten-plates` tasks.
The `bootstrap-*` tasks are curriculum subgoals for empty-inventory burner
startup. The `freeplay-burner-*` tasks use the vanilla Freeplay crashland
inventory and target additional iron plates above the starting `8`. The legacy
names are kept to preserve prior benchmark history. Burner-chain tasks require
enough ore to be produced by the burner miner before success is counted. CLI
commands accept `--task` and still allow low-level overrides such as
`--target-iron-plates` and `--max-steps` for quick experiments.

For reinforcement learning experiments, `NumericObservationWrapper` converts the
mock dictionary observation into a fixed numeric vector and drops the string
objective. This keeps the default environment readable while giving PPO a simple
`Box` observation space.

Recipe-realism and buffer changes add `iron_gear_wheel`, burner-chain production
fields, furnace buffers, and new discrete actions to numeric observations. PPO
models trained before these changes should be treated as historical artifacts
rather than compatible policies for current tasks.

The mock environment and numeric wrapper also expose `action_masks()`, returning
a NumPy boolean mask of valid discrete actions. Standard Stable-Baselines3 PPO
does not consume this mask directly; MaskablePPO from `sb3-contrib` uses it for
mask-aware training.

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
