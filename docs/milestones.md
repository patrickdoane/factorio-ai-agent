# Milestones

This file records durable project milestones and the commands needed to reproduce
them. Saved models under `/tmp/opencode` are local experiment artifacts; retrain
them with the listed commands when a clean machine needs the same policy.

## 2026-06-16: Miner Output To Furnace Input Transfer

The machine-local buffer model now supports an explicit direct transfer from a
burner miner output buffer into a furnace input buffer. The new
`TRANSFER_MINER_OUTPUT_TO_FURNACE` action is enabled only when miner output ore
exists, furnace input buffers are active, and a furnace is placed. This remains
an abstract direct transfer, not an inserter, belt, or spatial layout model.

Current best three-task miner-output/transfer policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-miner-transfer-multitask-50k.zip`
- Tasks: `buffered-miner-output-ore`, `buffered-miner-collect-ore`,
  `buffered-miner-transfer-plate`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `50000`

Aggregate benchmark result:

```text
score:              0.999300
success_rate:       1.000000
avg_steps:          7.000000
avg_reward:         13.263333
invalid_rate:       0.000000
eval_episodes:      60
```

Per-task benchmark results:

```text
buffered-miner-output-ore:     success_rate=1.000000 avg_steps=5.000000  invalid_rate=0.000000
buffered-miner-collect-ore:    success_rate=1.000000 avg_steps=6.000000  invalid_rate=0.000000
buffered-miner-transfer-plate: success_rate=1.000000 avg_steps=10.000000 invalid_rate=0.000000
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --tasks buffered-miner-output-ore,buffered-miner-collect-ore,buffered-miner-transfer-plate \
  --total-timesteps 50000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-miner-transfer-multitask-50k.zip
```

Aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-miner-transfer-multitask-50k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-miner-output-ore,buffered-miner-collect-ore,buffered-miner-transfer-plate \
  --eval-episodes 20 \
  --seed 1
```

Transfer specialist reference:

- Model path: `/tmp/opencode/maskable-ppo-buffered-miner-transfer-plate-50k.zip`
- Task: `buffered-miner-transfer-plate`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `50000`

Benchmark result:

```text
score:              0.999000
success_rate:       1.000000
avg_steps:          10.000000
avg_reward:         19.900000
invalid_rate:       0.000000
eval_episodes:      20
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-miner-transfer-plate \
  --total-timesteps 50000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-miner-transfer-plate-50k.zip
```

Benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-miner-transfer-plate-50k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-miner-transfer-plate \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The learned policy follows the 10-step sequence: place furnace, place burner
  miner, mine coal, insert fuel, wait, wait, transfer miner output to furnace,
  wait, wait, take furnace output.
- The task mask forces transfer when miner output is pending and forces furnace
  waiting while transferred furnace input is pending, preventing valid but
  off-task extra mining/refueling loops.

## 2026-06-16: Burner Miner Output Buffer Policies

The machine-local buffer model now includes burner miner output. The
`buffered-miner-*` tasks require placing and fueling a burner mining drill,
waiting for ore production, and optionally collecting that ore with
`TAKE_MINER_OUTPUT`. Older burner tasks still move miner output directly into
inventory unless `use_miner_output_buffer` is enabled.

Current best two-task miner-output policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-miner-multitask-20k.zip`
- Tasks: `buffered-miner-output-ore`, `buffered-miner-collect-ore`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `20000`

Aggregate benchmark result:

```text
score:              0.999450
success_rate:       1.000000
avg_steps:          5.500000
avg_reward:         9.945000
invalid_rate:       0.000000
eval_episodes:      40
```

Per-task benchmark results:

```text
buffered-miner-output-ore:  success_rate=1.000000 avg_steps=5.000000 invalid_rate=0.000000
buffered-miner-collect-ore: success_rate=1.000000 avg_steps=6.000000 invalid_rate=0.000000
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --tasks buffered-miner-output-ore,buffered-miner-collect-ore \
  --total-timesteps 20000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-miner-multitask-20k.zip
```

Aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-miner-multitask-20k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-miner-output-ore,buffered-miner-collect-ore \
  --eval-episodes 20 \
  --seed 1
```

Specialist policy references:

```text
buffered-miner-output-ore:  model=/tmp/opencode/maskable-ppo-buffered-miner-output-ore-10k.zip  success_rate=1.000000 avg_steps=5.000000 invalid_rate=0.000000
buffered-miner-collect-ore: model=/tmp/opencode/maskable-ppo-buffered-miner-collect-ore-10k.zip success_rate=1.000000 avg_steps=6.000000 invalid_rate=0.000000
```

Rollout quality note:

- `buffered-miner-output-ore` follows the 5-step sequence: place burner miner,
  mine coal, insert fuel, wait, wait.
- `buffered-miner-collect-ore` adds `TAKE_MINER_OUTPUT` as step 6 and ends with
  `iron_ore=1`, `miner_output_iron_ore=0`.
- This completes the source-side machine buffer needed before adding transfers
  from miner output to furnace input.

## 2026-06-16: Furnace Input Buffer Policies

The furnace buffer generation now includes explicit furnace input as well as
output. The `buffered-insert-*` tasks require the policy to mine ore, insert it
with `INSERT_IRON_ORE_INTO_FURNACE`, wait for smelting, and optionally collect
the result with `TAKE_FURNACE_OUTPUT`. Older buffer tasks keep their automatic
inventory-to-furnace input behavior.

Current best input-buffer specialists:

```text
buffered-insert-smelt-plate:          model=/tmp/opencode/maskable-ppo-buffered-insert-smelt-plate-10k.zip          success_rate=1.000000 avg_steps=5.000000  invalid_rate=0.000000
buffered-insert-collect-plate:        model=/tmp/opencode/maskable-ppo-buffered-insert-collect-plate-10k.zip        success_rate=1.000000 avg_steps=6.000000  invalid_rate=0.000000
buffered-insert-collect-three-plates: model=/tmp/opencode/maskable-ppo-buffered-insert-collect-three-mask-50k.zip   success_rate=1.000000 avg_steps=16.000000 invalid_rate=0.000000
```

Specialist benchmark results:

```text
buffered-insert-smelt-plate:
score:              0.999500
success_rate:       1.000000
avg_steps:          5.000000
avg_reward:         9.950000
invalid_rate:       0.000000
eval_episodes:      20

buffered-insert-collect-plate:
score:              0.999400
success_rate:       1.000000
avg_steps:          6.000000
avg_reward:         19.940000
invalid_rate:       0.000000
eval_episodes:      20

buffered-insert-collect-three-plates:
score:              0.998400
success_rate:       1.000000
avg_steps:          16.000000
avg_reward:         39.840000
invalid_rate:       0.000000
eval_episodes:      20
```

Training commands:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-insert-smelt-plate \
  --total-timesteps 10000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-insert-smelt-plate-10k.zip

factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-insert-collect-plate \
  --total-timesteps 10000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-insert-collect-plate-10k.zip

factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-insert-collect-three-plates \
  --total-timesteps 50000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-insert-collect-three-mask-50k.zip
```

Benchmark command examples:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-insert-collect-three-mask-50k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-insert-collect-three-plates \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The learned single-plate collection policy follows the 6-step sequence: place
  furnace, mine ore, insert ore, wait, wait, take furnace output.
- The learned repeated collection policy follows the 16-step repeated loop with
  three mine/insert/wait/wait/take cycles after furnace placement.
- The collection action mask now forces `TAKE_FURNACE_OUTPUT` while collection is
  the active success condition and furnace output is pending. This prevents
  policies from postponing collection by continuing to mine or insert ore.
- A 50k three-task input-buffer policy at
  `/tmp/opencode/maskable-ppo-buffered-insert-multitask-50k.zip` did not solve
  the aggregate curriculum (`success_rate=0.000000`, `invalid_rate=0.000000`),
  so multi-task input-buffer retention remains the next training problem.

## 2026-06-16: Furnace Output Buffer Policies

The next training generation starts with explicit furnace output buffer state and
explicit output collection. The `buffered-smelt-plate` task keeps smelted plates
in `furnace_output_iron_plate` instead of moving them directly into inventory,
while `buffered-collect-plate` requires the policy to collect that output with
`TAKE_FURNACE_OUTPUT`. Older tasks keep their inventory-output behavior by
default.

Current best three-task buffer policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-multitask-three-50k.zip`
- Tasks: `buffered-smelt-plate`, `buffered-collect-plate`,
  `buffered-collect-three-plates`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `50000`

Aggregate benchmark result:

```text
score:              0.999033
success_rate:       1.000000
avg_steps:          9.666667
avg_reward:         29.903333
invalid_rate:       0.000000
eval_episodes:      60
```

Per-task benchmark results:

```text
buffered-smelt-plate:          success_rate=1.000000 avg_steps=4.000000  invalid_rate=0.000000
buffered-collect-plate:        success_rate=1.000000 avg_steps=5.000000  invalid_rate=0.000000
buffered-collect-three-plates: success_rate=1.000000 avg_steps=20.000000 invalid_rate=0.000000
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --tasks buffered-smelt-plate,buffered-collect-plate,buffered-collect-three-plates \
  --total-timesteps 50000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-multitask-three-50k.zip
```

Aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-multitask-three-50k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-smelt-plate,buffered-collect-plate,buffered-collect-three-plates \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The three-task policy preserves solved behavior for smelting, single collection,
  and repeated collection with zero invalid actions.
- The repeated collection task completes one step slower than the single-task
  repeated policy, so the single-task model remains a slightly better specialist.

Previous two-task buffer policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-multitask-20k.zip`
- Tasks: `buffered-smelt-plate`, `buffered-collect-plate`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `20000`

Aggregate benchmark result:

```text
score:              0.999550
success_rate:       1.000000
avg_steps:          4.500000
avg_reward:         14.955000
invalid_rate:       0.000000
eval_episodes:      40
```

Per-task benchmark results:

```text
buffered-smelt-plate:   success_rate=1.000000 avg_steps=4.000000 invalid_rate=0.000000
buffered-collect-plate: success_rate=1.000000 avg_steps=5.000000 invalid_rate=0.000000
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --tasks buffered-smelt-plate,buffered-collect-plate \
  --total-timesteps 20000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-multitask-20k.zip
```

Aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-multitask-20k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-smelt-plate,buffered-collect-plate \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The multi-task policy preserves optimal behavior for both buffer tasks.
- It completes `buffered-smelt-plate` in 4 steps and `buffered-collect-plate` in
  5 steps with zero invalid actions.

Current best repeated buffered-collection policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-collect-three-50k.zip`
- Task: `buffered-collect-three-plates`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `50000`

Benchmark result:

```text
score:              0.998100
success_rate:       1.000000
avg_steps:          19.000000
avg_reward:         59.810000
invalid_rate:       0.000000
eval_episodes:      20
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-collect-three-plates \
  --total-timesteps 50000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-collect-three-50k.zip
```

Benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-collect-three-50k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-collect-three-plates \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The learned policy solves repeated furnace-output collection with zero invalid
  actions.
- It finishes in 19 steps by sometimes letting the furnace output buffer build up
  before collection; the scripted shortest repeated loop is 13 steps, so future
  optimization can target earlier output collection rather than semantics changes.

Current best buffered-collection policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-collect-plate-10k.zip`
- Task: `buffered-collect-plate`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `10000`

Benchmark result:

```text
score:              0.999500
success_rate:       1.000000
avg_steps:          5.000000
avg_reward:         19.950000
invalid_rate:       0.000000
eval_episodes:      20
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-collect-plate \
  --total-timesteps 10000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-collect-plate-10k.zip
```

Benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-collect-plate-10k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-collect-plate \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The learned policy follows the optimal 5-step sequence: place furnace, mine ore,
  wait, wait, take furnace output.
- Final state has `iron_plate=1` and `furnace_output_iron_plate=0`, confirming
  success requires possession rather than only machine-local output.

Current best buffered-smelting policy:

- Model path: `/tmp/opencode/maskable-ppo-buffered-smelt-plate-10k.zip`
- Task: `buffered-smelt-plate`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Training steps: `10000`

Benchmark result:

```text
score:              0.999600
success_rate:       1.000000
avg_steps:          4.000000
avg_reward:         9.960000
invalid_rate:       0.000000
eval_episodes:      20
```

Training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --task buffered-smelt-plate \
  --total-timesteps 10000 \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-buffered-smelt-plate-10k.zip
```

Benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-buffered-smelt-plate-10k.zip \
  --model-algo maskable-ppo \
  --tasks buffered-smelt-plate \
  --eval-episodes 20 \
  --seed 1
```

Rollout quality note:

- The learned policy follows the optimal 4-step sequence: place furnace, mine ore,
  wait, wait.
- Final state has `iron_plate=0` and `furnace_output_iron_plate=1`, confirming
  success is driven by machine-local output state rather than inventory.

## 2026-06-16: Empty-Inventory Burner Bootstrap

The mock environment now has a learned policy that starts from empty inventory,
crafts and places the first furnace, manually smelts the materials needed for a
burner mining drill, places and fuels that drill, and completes the current
empty-inventory `burner-*` plate tasks with burner-mined ore required for
success.

This milestone is still abstract: production uses inventory counters and
wait-based timers, not spatial output directions, inserters, belts, or furnace
output buffers. The important capability step is empty-inventory bootstrap, not
logistics realism.

Current best empty-bootstrap policy:

- Model path: `/tmp/opencode/maskable-ppo-burner-multitask-empty-bootstrap-150k.zip`
- Base model: `/tmp/opencode/maskable-ppo-burner-first-from-smeltmask-coalcap-100k.zip`
- Fine-tuning tasks: `burner-first-plate`, `burner-three-plates`, `burner-ten-plates`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Seed: `42`
- Fine-tuning steps: `150000`

Aggregate benchmark result:

```text
score:              0.993100
success_rate:       1.000000
avg_steps:          69.000000
avg_reward:         89.310000
invalid_rate:       0.000000
eval_episodes:      30
```

Per-task benchmark results:

```text
burner-first-plate:  success_rate=1.000000 avg_steps=50.000000 invalid_rate=0.000000
burner-three-plates: success_rate=1.000000 avg_steps=58.000000 invalid_rate=0.000000
burner-ten-plates:   success_rate=1.000000 avg_steps=99.000000 invalid_rate=0.000000
```

Fine-tuning command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --tasks burner-first-plate,burner-three-plates,burner-ten-plates \
  --load-path /tmp/opencode/maskable-ppo-burner-first-from-smeltmask-coalcap-100k.zip \
  --total-timesteps 150000 \
  --n-steps 128 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --seed 42 \
  --device cpu \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-burner-multitask-empty-bootstrap-150k.zip \
  --eval-episodes 10
```

Aggregate benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-burner-multitask-empty-bootstrap-150k.zip \
  --model-algo maskable-ppo \
  --tasks burner-first-plate,burner-three-plates,burner-ten-plates \
  --eval-episodes 10 \
  --seed 42 \
  --append-results
```

Rollout inspection command:

```bash
factorio-ai run-ppo \
  --model-path /tmp/opencode/maskable-ppo-burner-multitask-empty-bootstrap-150k.zip \
  --model-algo maskable-ppo \
  --task burner-ten-plates \
  --seed 42 \
  --max-steps 180
```

The solved `burner-ten-plates` rollout completes in 99 steps with zero invalid
actions. It mines stone, crafts and places a furnace, manually smelts enough
plates for gears and drill crafting, crafts a second furnace for the burner-drill
recipe, places and fuels the drill, then runs the burner miner until the
burner-mined ore success requirement is met. The rollout is correct but not yet
efficient: it manually smelts surplus plates before committing to the drill, so
future optimization can target earlier drill construction and fewer manual-smelt
cycles without changing the success semantics.

The policy depended on explicit curriculum and task-aware masks. Important mask
fixes closed valid but off-task loops: extra furnace placement, no-op waits, and
excess pre-drill coal mining.

The previous single-task empty-bootstrap policy solved only `burner-first-plate`
and did not transfer to `burner-three-plates` or `burner-ten-plates`:

- Model path: `/tmp/opencode/maskable-ppo-burner-first-from-smeltmask-coalcap-100k.zip`
- Base model: `/tmp/opencode/maskable-ppo-bootstrap-curriculum-smeltmask-100k.zip`
- Fine-tuning task: `burner-first-plate`
- Benchmark: `success_rate=1.000000`, `avg_steps=51.000000`, `invalid_rate=0.000000`

Solved bootstrap curriculum policy:

- Model path: `/tmp/opencode/maskable-ppo-bootstrap-curriculum-smeltmask-100k.zip`
- Tasks: `bootstrap-craft-furnace`, `bootstrap-smelt-plates`, `bootstrap-craft-drill`, `bootstrap-place-and-fuel-drill`
- Algorithm: MaskablePPO
- Reward shaping: `burner-progress`
- Seed: `42`
- Training steps: `100000`

Aggregate curriculum benchmark result:

```text
score:              0.998975
success_rate:       1.000000
avg_steps:          10.250000
avg_reward:         29.897500
invalid_rate:       0.000000
eval_episodes:      40
```

Curriculum training command:

```bash
factorio-ai train-ppo \
  --algo maskable-ppo \
  --tasks bootstrap-craft-furnace,bootstrap-smelt-plates,bootstrap-craft-drill,bootstrap-place-and-fuel-drill \
  --total-timesteps 100000 \
  --n-steps 128 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --seed 42 \
  --device cpu \
  --reward-shaping burner-progress \
  --save-path /tmp/opencode/maskable-ppo-bootstrap-curriculum-smeltmask-100k.zip \
  --eval-episodes 10
```

Curriculum benchmark command:

```bash
factorio-ai research-benchmark \
  --agent ppo \
  --model-path /tmp/opencode/maskable-ppo-bootstrap-curriculum-smeltmask-100k.zip \
  --model-algo maskable-ppo \
  --tasks bootstrap-craft-furnace,bootstrap-smelt-plates,bootstrap-craft-drill,bootstrap-place-and-fuel-drill \
  --eval-episodes 10 \
  --seed 42 \
  --append-results
```

Recommended next research target: improve empty-bootstrap efficiency on
`burner-ten-plates` by encouraging earlier drill construction and fewer manual
smelt cycles, or move the mock environment toward richer automation mechanics
such as output buffers and simple inserter/belt abstractions.

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
