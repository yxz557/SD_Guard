# Nothink + public AgentDojo v1.2.1

This pipeline keeps Qwen3-8B nothink as the target domain and uses only the
explicitly versioned, no-defense public AgentDojo traces as training
augmentation. Think traces are not included.

The main dataset policy is:

- group key: `(suite_name, user_task_id, injection_task_id)`;
- reuse the frozen nothink split manifest;
- main train: nothink train + public traces whose task pair is train;
- main validation/test: nothink only;
- public validation/test pairs: auxiliary files only, never main train;
- exact serialized duplicates are removed, with local nothink preferred;
- incompatible Qwen feature configurations fail fast.

## 1. Verify the two existing local paths

```bash
export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export FORMAL_EXP="$SD_ROOT/Code/FormalExp"
export LOCAL_TRACE_ROOT="$SD_ROOT/results/agentdojo/task_injection_nothink_ctx40k_r1"
export LOCAL_FEATURE_ROOT="$SD_ROOT/results/formal/guard_training/qwen3_4b_task_injection_nothink_ctx40k_r1"

test -f "$LOCAL_TRACE_ROOT/analysis/trace_records.csv"
find "$LOCAL_TRACE_ROOT/traces" -name raw_trace.json | wc -l
find "$LOCAL_TRACE_ROOT/traces" -name ground_truth.json | wc -l

test -f "$LOCAL_FEATURE_ROOT/feature_config.json"
test -f "$LOCAL_FEATURE_ROOT/split_manifest.jsonl"
find "$LOCAL_FEATURE_ROOT/features" -name 'shard_*.jsonl' -print0 \
  | xargs -0 cat \
  | wc -l
```

The trace root is the complete raw trajectory dataset. The feature root is the
Qwen3-4B prefill output, including feature shards and the existing split.

## 2. Check whether public runs are already downloaded

```bash
find /root/autodl-tmp -type d \
  -path '*/runs/Meta-SecAlign-70B' \
  -print
```

If this prints a path, its parent directory is a candidate public `runs` root.
Verify it before reusing it:

```bash
export PUBLIC_RUNS=/path/printed/above/..
find "$PUBLIC_RUNS" -name '*.json' | wc -l
test -d "$PUBLIC_RUNS/meta-llama_Llama-3.3-70B-Instruct"
test -d "$PUBLIC_RUNS/Meta-SecAlign-70B"
```

The current official snapshot contains roughly 36k JSON files. A much smaller
count usually means this is not the full public runs directory.

## 3. Download the official public runs when absent

This is a GitHub download; `HF_ENDPOINT` and Hugging Face offline flags do not
affect it.

```bash
export PUBLIC_REPO=/root/autodl-tmp/datasets/agentdojo-public
mkdir -p "$(dirname "$PUBLIC_REPO")"

git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/ethz-spylab/agentdojo.git \
  "$PUBLIC_REPO"

git -C "$PUBLIC_REPO" sparse-checkout set runs
git -C "$PUBLIC_REPO" rev-parse HEAD \
  | tee "$PUBLIC_REPO/DOWNLOADED_COMMIT.txt"

export PUBLIC_RUNS="$PUBLIC_REPO/runs"
find "$PUBLIC_RUNS" -name '*.json' | wc -l
```

Do not run `git clone` again when `$PUBLIC_REPO/.git` already exists. In that
case, set `PUBLIC_RUNS` to the existing checkout.

## 4. Convert the clean public subset to the local trace schema

The defaults retain only:

- `benchmark_version == v1.2.1`;
- no defense;
- no run error;
- Boolean AgentDojo security label;
- `direct`, `ignore_previous`, and `important_instructions`.

```bash
export PUBLIC_TRACE_ROOT="$SD_ROOT/results/agentdojo_public/v1_2_1_no_defense"

cd "$FORMAL_EXP"

python -m guard_training.public_agentdojo.prepare_traces \
  --runs-root "$PUBLIC_RUNS" \
  --output-root "$PUBLIC_TRACE_ROOT" \
  --benchmark-version v1.2.1 \
  --defense none \
  --attacks direct ignore_previous important_instructions
```

For the currently published snapshot, the expected pre-feature-extraction
inventory is 5,694 attacked traces, 415 positives, 5,279 negatives, and 949 task
pairs.

## 5. Prefill public traces with exactly the same Qwen3-4B setup

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd-guard-features-cu128

export QWEN4B=/root/autodl-tmp/models/Qwen3-4B
export PUBLIC_FEATURE_ROOT="$SD_ROOT/results/formal/guard_training/qwen3_4b_agentdojo_public_v1_2_1_no_defense"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd "$FORMAL_EXP"

python -m guard_training.qwen3_trace_features.extract_features \
  --trace-root "$PUBLIC_TRACE_ROOT" \
  --output-dir "$PUBLIC_FEATURE_ROOT" \
  --model-path "$QWEN4B" \
  --model-dtype bfloat16 \
  --storage-dtype float16 \
  --attn-implementation sdpa \
  --shard-size 128 \
  --continue-on-error

python -m guard_training.qwen3_trace_features.inspect_features \
  --feature-root "$PUBLIC_FEATURE_ROOT" \
  --check-finite \
  --output "$PUBLIC_FEATURE_ROOT/feature_report.json"
```

The mixed builder compares the public and local `feature_config.json` files and
stops if the Qwen model, tokenizer, serialization, saved layers, context length,
or readout setup differs.

## 6. Build the compact mixed offline dataset

```bash
export LOCAL_MANIFEST="$LOCAL_FEATURE_ROOT/split_manifest.jsonl"
export MIXED_DATASET="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/offline_datasets/trace_last_layer_final"

python -m guard_training.public_agentdojo.build_mixed_offline_dataset \
  --local-feature-root "$LOCAL_FEATURE_ROOT" \
  --public-feature-root "$PUBLIC_FEATURE_ROOT" \
  --local-manifest "$LOCAL_MANIFEST" \
  --output-dir "$MIXED_DATASET" \
  --feature trace_last \
  --layer -1 \
  --storage-dtype float16 \
  --unmapped-public exclude
```

Inspect these files before training:

```bash
cat "$MIXED_DATASET/split_summary.json"
cat "$MIXED_DATASET/dataset_config.json"
wc -l "$MIXED_DATASET"/*.metadata.jsonl
wc -l "$MIXED_DATASET/excluded.jsonl"
```

## 7. Train and evaluate 10 seeds

```bash
export REPEATED_ROOT="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/experiments/trace_last_final_mlp_10seeds"

python -m guard_training.frozen_hidden_mlp.run_repeated \
  --offline-dataset "$MIXED_DATASET" \
  --output-root "$REPEATED_ROOT" \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --head mlp \
  --hidden-dim 256 \
  --dropout 0.1 \
  --positive-weight inverse \
  --learning-rate 1e-3 \
  --weight-decay 1e-2 \
  --batch-size 128 \
  --epochs 50 \
  --patience 5 \
  --device auto
```

`run_repeated` selects the threshold on the local nothink validation split and
reports the main metrics on the local nothink test split. The compact dataset
also contains `public_validation` and `public_test` for separate cross-model
evaluation; they are not mixed into the headline test metric.

