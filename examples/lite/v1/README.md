# Lite v1 — desktop.use consolidated training walkthrough

## Desktop

Train one model on **several desktop.use datasets at once** by concatenating them into a single
SFT parquet — here **Lite.ScaleCUA + Lite.CUAGym**; add more as they land. They share the same
`qwen3_5` reasoning adapter and identical `agent_kwargs`, so one config drives the export.
Lite.ScaleCUA is far larger, so we cap it to 5000 train trajectories (as in
[README.md](/README.md#sft-any-cua-on-any-datasets)) and concatenate that with the full
Lite.CUAGym set.

### SFT

#### Export

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
OUT=.data/sft/qwen3_5-reasoning/desktop.use

# 1. download both datasets into the local dataset cache
uv run python -m lite.data.hf.download Lite.ScaleCUA --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA"
uv run python -m lite.data.hf.download Lite.CUAGym   --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym"

# 2. internalize CoT: the config reasons in Qwen3.5's native <think> channel, but the teacher
#    saved its Thought as an inline_reasoning part. Move it into reasoning_content, which the
#    chat template renders as <think>. Images stay path-referenced, so the .think copy is tiny.
for ds in Lite.ScaleCUA Lite.CUAGym; do
  uv run python examples/lite/v1/internalize_cot.py \
    --in "${CUA_LITE_DATASETS_ROOT}/cua-lite/${ds}" --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/${ds}.think"
done

# 3. export one SFT parquet per dataset (--image-root = the dir ABOVE cua-lite/). --filter keeps
#    the clean, successful trajectories. --sample is GLOBAL over --data-paths, so Lite.ScaleCUA is
#    exported on its own to cap only it.
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/desktop.use.compact.reasoning.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.think" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "$FILTER" --sample 5000 --seed 42 \
  -o "$OUT/scalecua_5k.parquet"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/desktop.use.compact.reasoning.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym.think" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "$FILTER" \
  -o "$OUT/cuagym.parquet"

# 4. concat the two into one training set
uv run python -m lite.data.merge -i "$OUT/cuagym.parquet" "$OUT/scalecua_5k.parquet" -o "$OUT/train.parquet"
```

#### Train

```bash
# --- Slime container ---
# Same SFT recipe, pointed at the desktop.use parquet. SFT at TP=2 (8 GPUs → DP=4).
# BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack). One ckpt per epoch.
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/desktop.use/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/desktop.use/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval

Base model vs SFT checkpoint on the full `lite.osworld` eval split (332 scored tasks after
`--filter`) — the [Lite.OSWorld row](/docs/eval.md#osworld--liteosworld) of
[docs/eval.md](/docs/eval.md). Env setup:
[`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-reasoning-4b/desktop.use/sft/iter_<N> \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.reasoning.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft &

wait
```

Score each run from `.logs/rollout/<model_slug>/<env_id>/<role>/summary.json` →
`stats.mean_episode_return` (denominator is `num_valid`).

<!--
## Ablations

The reasoning ablation: train a second checkpoint on the same trajectories with the same
recipe, only swapping `desktop.use.compact.reasoning.yaml` for `desktop.use.compact.yaml`
(Action-only, no `<think>` channel). Everything below mirrors [SFT](#sft) step for step.

One flow difference, and it is forced: `desktop.use.compact.yaml` leaves `enable_thinking` off
(the adapter default), so it must export from the ORIGINAL datasets, not the `.think` copies.
Exporting `.think` data with thinking off fails fast — the internalized reasoning lives in the
target, so the prompt/target boundary no longer holds (`SFT prompt/target boundary broke
(enable_thinking=False)`). So there is no internalize-CoT step here.

### SFT (Action-only)

#### Export

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
OUT=.data/sft/qwen3_5/desktop.use

# 1. export one SFT parquet per dataset (--image-root = the dir ABOVE cua-lite/). --filter keeps
#    the clean, successful trajectories. --sample is GLOBAL over --data-paths, so Lite.ScaleCUA is
#    exported on its own to cap only it. Both datasets are already in the cache from the SFT flow
#    above; these are the ORIGINAL trees, not the .think copies.
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "$FILTER" --sample 5000 --seed 42 \
  -o "$OUT/scalecua_5k.parquet"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "$FILTER" \
  -o "$OUT/cuagym.parquet"

# 2. concat the two into one training set
uv run python -m lite.data.merge -i "$OUT/cuagym.parquet" "$OUT/scalecua_5k.parquet" -o "$OUT/train.parquet"
```

#### Train

```bash
# --- Slime container ---
# Same SFT recipe, pointed at the Action-only desktop.use parquet. SFT at TP=2 (8 GPUs → DP=4).
# BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack). One ckpt per epoch.
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5/desktop.use/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-4b/desktop.use/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval

Base model vs SFT checkpoint on the full `lite.osworld` eval split (332 scored tasks after
`--filter`) — the [Lite.OSWorld row](/docs/eval.md#osworld--liteosworld) of
[docs/eval.md](/docs/eval.md). Env setup:
[`lite/gym/envs/lite/osworld/README.md`](/lite/gym/envs/lite/osworld/README.md).

```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-4b/desktop.use/sft/iter_<N> \
  --env-id lite.osworld --splits eval --concurrency 8 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/desktop.use.compact.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft &

wait
```

Compare `stats.mean_episode_return` against the reasoning checkpoint from [Eval](#eval). Both runs
use the same tasks, seeds, and step budget, so the only moving part is the `<think>` channel.
Do not compare runs that use different task sets, sampling configurations, or task budgets.
-->
