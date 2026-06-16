# Milestones

This file records durable project milestones and the commands needed to reproduce
them. Saved models under `/tmp/opencode` are local experiment artifacts; retrain
them with the listed commands when a clean machine needs the same policy.

## 2026-06-15: Abstract Freeplay Burner Policy

The mock environment is still intentionally abstract: production is represented
with counters and wait-based timers, not spatial output directions, inserters,
belts, or furnace output buffers.

Current best policy:

- Model path: `/tmp/opencode/maskable-ppo-freeplay-multitask-finetuned-25k.zip`
- Base model: `/tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip`
- Fine-tuning tasks: `freeplay-burner-first-plate`, `freeplay-burner-three-plates`, `freeplay-burner-ten-plates`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Seed: `42`
- Fine-tuning steps: `25000`

Aggregate benchmark result:

```text
score:              0.997833
success_rate:       1.000000
avg_steps:          21.666667
avg_reward:         46.450000
invalid_rate:       0.000000
eval_episodes:      30
```

Fine-tuning command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --load-path /tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip \
  --tasks freeplay-burner-first-plate,freeplay-burner-three-plates,freeplay-burner-ten-plates \
  --total-timesteps 25000 \
  --n-steps 256 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --seed 42 \
  --device cpu \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-freeplay-multitask-finetuned-25k.zip \
  --eval-episodes 5
```

Current aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-freeplay-multitask-finetuned-25k.zip \
  --model-algo maskable-ppo \
  --tasks freeplay-burner-first-plate,freeplay-burner-three-plates,freeplay-burner-ten-plates \
  --eval-episodes 10 \
  --seed 42 \
  --append-results
```

The previous best single-task-trained policy was:

- Model path: `/tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip`
- Training task: `freeplay-burner-ten-plates`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Seed: `42`
- Training steps: `50000`

The ten-plate policy generalizes backward across the full current Freeplay burner
set:

- `freeplay-burner-first-plate`
- `freeplay-burner-three-plates`
- `freeplay-burner-ten-plates`

Previous aggregate benchmark result:

```text
score:              0.997167
success_rate:       1.000000
avg_steps:          28.333333
avg_reward:         46.383333
invalid_rate:       0.000000
eval_episodes:      30
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --task freeplay-burner-ten-plates \
  --total-timesteps 50000 \
  --n-steps 256 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --seed 42 \
  --device cpu \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip \
  --eval-episodes 5
```

Rollout inspection command:

```bash
factorio-ai run-ppo \
  --model-path /tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip \
  --model-algo maskable-ppo \
  --task freeplay-burner-ten-plates \
  --seed 42 \
  --max-steps 70
```

Previous aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-freeplay-burner-ten-refuel-capped-50k.zip \
  --model-algo maskable-ppo \
  --tasks freeplay-burner-first-plate,freeplay-burner-three-plates,freeplay-burner-ten-plates \
  --eval-episodes 10 \
  --seed 42 \
  --append-results
```

The successful rollout queues enough coal, waits through burner mining and
smelting, and completes without invalid actions or gear crafting. It may still
queue a small amount of extra fuel on the ten-plate task, so future efficiency
work can target fewer steps without changing success semantics.

Recommended next research target: move to the empty-inventory `burner-*`
bootstrap tasks before adding symbolic logistics such as miner output direction
or furnace output buffers.
