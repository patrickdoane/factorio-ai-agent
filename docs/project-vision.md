# Project Vision

This document preserves the long-term intent of `factorio-ai-agent` for future
LLM sessions, provider switches, and development modes.

## Long-Term Goal

Build AI agents that can autonomously play Factorio, progressing from controlled
mock production tasks to real-game rocket launch and eventually speedrun-style
optimization.

Rocket launch is the first major endgame capability milestone. Speedrun-style
play is the optimization frontier after competence: minimizing time-to-objective,
choosing efficient build orders, routing actions well, and adapting to map and
resource constraints.

## Current Approach

The project starts with a small deterministic mock environment instead of real
Factorio. This is intentional. The mock environment gives us a stable place to
build observations, actions, agents, benchmarks, training loops, and result logs
before adding the complexity of a real Factorio process, mod bridge, reset
harness, and scenario runner.

Current benchmark tasks are small burner-era objectives such as producing one,
three, or ten iron plates. These are primitive compared with Factorio, but they
exercise the first important skills: mine, craft, place, fuel, wait for
production, and complete an objective.

## Capability Ladder

1. Primitive competence: mine resources, craft items, place entities, fuel early
   machines, wait for production, and avoid invalid actions.
2. Automation competence: build stable mining and smelting loops that sustain
   production instead of only completing one-off recipes.
3. Planning competence: decompose goals into subgoals, identify missing
   infrastructure, recover from bottlenecks, and choose among alternative build
   orders.
4. Real Factorio competence: control a real Factorio scenario through a bounded
   adapter with structured observations, safe actions, reset support, timeouts,
   and reproducible scoring.
5. Rocket launch competence: progress through science, technologies, oil,
   circuits, modules, rocket parts, and launch a rocket reliably.
6. Speedrun optimization: improve time-to-rocket or time-to-objective through
   better sequencing, routing, throughput, and strategic tradeoffs.

## Benchmark Philosophy

Benchmarks exist to measure agent behavior, not to become the behavior. Keep task
definitions, scoring semantics, and success criteria stable during research so
score changes mean the agent improved.

The main benchmark fields are:

- `score`: primary scalar to optimize and compare.
- `success_rate`: whether the agent solves the task.
- `avg_steps`: how efficiently it solves or attempts the task.
- `avg_reward`: diagnostic signal for progress and learning behavior.
- `invalid_rate`: whether the policy is attempting impossible actions.
- `eval_episodes`: sample size for the result.

The random baseline is a weak sanity check. The scripted baseline is a strong
hand-written reference. Learned agents should move from random-like metrics
toward scripted-like metrics, then eventually exceed scripted behavior on tasks
where optimization matters.

## Agent Types

- Random agents establish weak lower-bound behavior and reproducibility checks.
- Scripted agents validate environment mechanics and provide strong references.
- Learned agents, such as PPO policies, should improve through training and
  evaluation cycles rather than hard-coded task-specific scripts.
- Future planner or hybrid agents may combine search, heuristics, learned
  policies, and structured game state.

## Research Guardrails

- Do not improve benchmark scores by changing tasks, rewards, success criteria,
  or benchmark scoring.
- Keep the editable research surface narrow during autonomous loops.
- Log experiment results, including regressions and crashes.
- Prefer small, reversible changes with clear benchmark evidence.
- Validate real Factorio integration separately from model quality.

## Near-Term Roadmap

1. Keep the mock benchmark deterministic and reproducible.
2. Improve training and observation code against fixed mock tasks.
3. Add curriculum or mask-aware learning if it improves fixed benchmark results.
4. Expand mock tasks toward richer production chains without invalidating prior
   benchmark comparisons.
5. Build a read-only real Factorio adapter.
6. Add bounded real actions and scenario reset support.
7. Run scripted primitive tasks in real Factorio.
8. Train or evaluate learned and hybrid agents against real scenarios.
9. Establish rocket-launch and time-to-rocket benchmarks.
