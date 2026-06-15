# Autonomous Research Program

Use this file as the human-maintained instruction sheet for autoresearch-style
experiments in `factorio-ai-agent`.

## Goal

Improve agent performance on fixed mock Factorio benchmarks without changing the
benchmark, task definitions, or success criteria.

## Benchmark Command

Run the fixed benchmark with:

```bash
factorio-ai research-benchmark --agent scripted --tasks first-plate,three-plates --eval-episodes 10 --seed 42
```

For random baseline checks:

```bash
factorio-ai research-benchmark --agent random --tasks first-plate,three-plates --eval-episodes 10 --seed 42
```

The command emits a final machine-readable summary block:

```text
---
score:              0.000000
success_rate:       0.000000
avg_steps:          0.000000
avg_reward:         0.000000
invalid_rate:       0.000000
eval_episodes:      0
```

Optimize `score`. Use `success_rate`, `avg_steps`, `avg_reward`, and
`invalid_rate` as diagnostics.

## Editable Surface

Allowed for autonomous research:

- `src/factorio_ai_agent/training/train_ppo.py`
- `src/factorio_ai_agent/envs/numeric_observation_wrapper.py`
- new files under `src/factorio_ai_agent/research/` that do not alter benchmark
  scoring semantics

Read-only during autonomous research:

- `src/factorio_ai_agent/tasks.py`
- `src/factorio_ai_agent/envs/mock_factorio_env.py`
- `src/factorio_ai_agent/research/benchmark.py`
- tests defining benchmark expectations

## Logging

Use untracked `results.tsv` for experiment notes. Record at least:

- timestamp
- git commit
- command
- score
- success_rate
- avg_steps
- avg_reward
- invalid_rate
- notes

## Guardrails

- Do not change tasks, rewards, success criteria, or benchmark scoring to improve
  results.
- Keep changes small and reversible.
- Run `make test` before keeping a change.
- Discard regressions unless they reveal a benchmark bug.
