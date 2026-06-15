# Autoresearch Handoff

This document summarizes the current discussion about using Andrej Karpathy's experimental `autoresearch` workflow with `factorio-ai-agent`, and suggests concrete next steps for continuing from another device or session.

## Current Project State

`factorio-ai-agent` is currently a prototype framework for small Factorio-like tasks. It already includes:

- A deterministic `MockFactorioEnv` with a Gymnasium-style interface.
- Named tasks: `first-plate`, `three-plates`, and `ten-plates`.
- Baseline agents: random and scripted burner-miner agents.
- A numeric observation wrapper for reinforcement learning.
- An optional Stable-Baselines3 PPO training entry point.
- Evaluation helpers and CLI commands for running and comparing agents.
- Action masks exposed by the mock environment and numeric wrapper.

The current repository is ready for controlled mock-environment experiments, but it is not yet ready for reliable autonomous model improvement or real Factorio play. The missing pieces are a fixed benchmark harness, a narrow research protocol, and eventually a real Factorio adapter.

## Autoresearch Takeaways

The `autoresearch` repository is intentionally simple. Its useful pattern is not a dependency so much as a workflow:

- A human-maintained `program.md` gives the agent operating instructions.
- The agent modifies one narrow research surface.
- The agent runs one fixed-budget experiment command.
- The experiment emits one comparable scalar metric.
- The agent commits improvements and discards regressions.
- Every experiment is logged to an untracked `results.tsv`.

For this project, the right adaptation is to build an autoresearch-style workflow inside `factorio-ai-agent`, not to vendor or copy the upstream repository wholesale.

## Recommended Local Research Shape

Add a small research mode around the existing mock environment and PPO stack.

Proposed files or commands:

- `program.md`: human-maintained autonomous research instructions.
- `src/factorio_ai_agent/research/benchmark.py`: fixed benchmark harness.
- `factorio-ai research-benchmark`: CLI entry point for the fixed benchmark.
- `results.tsv`: untracked experiment log created by the agent during runs.
- `Makefile` target such as `make research-benchmark`.

The benchmark should emit a final machine-readable summary like:

```text
---
score:              0.873500
success_rate:       0.950000
avg_steps:          42.300000
avg_reward:         18.250000
invalid_rate:       0.010000
train_seconds:      300.0
eval_episodes:      100
```

The first scalar metric can be simple. For example, maximize `success_rate`, with `avg_steps` and `invalid_rate` as tie-breakers. A single `score` field is useful because autonomous agents need an unambiguous keep/discard rule.

## Suggested Editable Surface

For the first autonomous research loop, keep the allowed edit surface narrow:

- Editable: `src/factorio_ai_agent/training/train_ppo.py`.
- Editable: `src/factorio_ai_agent/envs/numeric_observation_wrapper.py`.
- Optional later: a dedicated research agent or PPO config module.
- Read-only during research: `tasks.py`, `mock_factorio_env.py`, and benchmark/evaluation logic.

This prevents metric hacking. If an agent can change task definitions, success criteria, or the benchmark itself, it can improve the score without improving the model.

## Promising Experiment Tracks

Start with the lowest-risk tracks first:

- PPO hyperparameter tuning: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `ent_coef`, and policy architecture.
- Observation representation: normalized counts, remaining step budget, target progress ratios, current valid-action mask features, and objective/state encodings.
- Curriculum training: train on `first-plate`, then `three-plates`, then `ten-plates`; evaluate only on fixed benchmark tasks.
- Mask-aware RL: evaluate whether adding `sb3-contrib` and `MaskablePPO` is worth the dependency cost because the environment already exposes action masks.
- Planner or scripted-agent research: add a dedicated `ResearchAgent` and benchmark it across task variations before investing too much in PPO.

Avoid reward-shaping research until the benchmark is fixed and guarded. Reward changes can be useful, but they are also the easiest way to create misleading progress.

## Real Factorio Readiness

The agent can attempt to play a real Factorio session once three components exist:

- A real Factorio adapter can read structured game state.
- A bounded action interface can issue safe actions.
- A scenario harness can reset the world and measure success.

Do not wait for a strong RL model before the first real-game test. The first live Factorio player should be a scripted or planner agent, because that validates the bridge independently of learning quality.

Recommended readiness levels:

- Level 0: Mock only. This is the current state.
- Level 1: Real Factorio read-only adapter. The agent can observe inventory, entities, resources, recipes, surface/chunk state, and game tick.
- Level 2: Scripted real-game control. The agent can move, mine, craft, place, insert fuel/items, wait, and query state. This is the first point where it can meaningfully try to play.
- Level 3: Real scenario reset and benchmark. Fixed save, automatic reset, max tick/time budget, logs, replays, and process recovery.
- Level 4: Learned policy against the real adapter. PPO or another learned policy controls the game through the same bounded action API.

A small Factorio mod bridge is probably better than pure RCON for this project because it can expose structured observations and constrained commands. RCON can still be useful for orchestration.

## Suggested Next Steps

1. Add `factorio-ai research-benchmark` using the mock environment.
2. Make the benchmark deterministic with fixed seeds, fixed task set, and fixed eval episode counts.
3. Emit a final summary containing `score`, `success_rate`, `avg_steps`, `avg_reward`, and `invalid_rate`.
4. Add `program.md` describing the autonomous loop and the allowed edit surface.
5. Run a baseline benchmark and record it in an untracked `results.tsv`.
6. Let the agent run short PPO/observation experiments and keep only score-improving changes.
7. Add a read-only real Factorio adapter spike.
8. Add bounded real actions and run the scripted first-plate task against a fixed save.
9. Only after the real adapter is repeatable, use autoresearch-style loops against real Factorio scenarios.

## Useful Commands

Current commands from the repository:

```bash
make test
make evaluate
make train-ppo
factorio-ai list-tasks
factorio-ai evaluate --agent both --task three-plates --episodes 10 --seed 42
factorio-ai train-ppo --task first-plate --total-timesteps 256 --n-steps 64 --batch-size 32 --eval-episodes 3
```

The future research loop should eventually look like:

```bash
factorio-ai research-benchmark --train-seconds 300 --eval-episodes 100 --seed 42
```

## Important Guardrails

- Keep benchmark, task definitions, and real-game success criteria read-only during autonomous research.
- Prefer simple improvements over complex changes with tiny metric gains.
- Log crashes and regressions instead of hiding them.
- Do not run autonomous loops against real Factorio until reset and timeout handling are reliable.
- Treat real Factorio integration bugs separately from model-quality experiments.
