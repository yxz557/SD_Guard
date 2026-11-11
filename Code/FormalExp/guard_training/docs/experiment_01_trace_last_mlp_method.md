# Guard Training 实验 01：Qwen3-4B Trace-Last Hidden State + MLP

## 1. 实验目的

本实验验证一个最简单的完整 trace 分类方案是否可行：

```text
AgentDojo task + 完整行为 trace
    -> Qwen3-4B 只做一次 forward/prefill
    -> 最后一个 trace token 的最终层 hidden state
    -> 冻结特征上的小型 MLP
    -> 攻击成功 / 未攻击成功
```

实验标签为 AgentDojo `ground_truth.json` 中的 `hijacked`：

- `1`：攻击成功；
- `0`：攻击未成功。

最终目标是对完整 trace 做 post-hoc 二分类。本实验不是在工具调用执行前进行拦截的在线 Guard。

## 2. 实验身份

| 项目 | 配置 |
|---|---|
| 实验简称 | `experiment_01_trace_last_mlp` |
| 原始 trace 生成模型 | Qwen3-8B |
| 特征 backbone | Qwen3-4B |
| backbone 是否训练 | 否，只做离线 forward |
| 第一版输入特征 | `trace_last`, `layer=-1` |
| 分类头 | 256 维单隐藏层 MLP |
| 标签 | `hijacked` |
| 随机种子 | `42` 至 `51`，共 10 次 |
| 数据划分 seed | `20261106` |
| 划分比例 | 70% / 15% / 15% |

原始 Qwen3-8B trace 的生成参数、vLLM 命令、thinking 和采样参数说明见：

[`qwen3_8b_task_injection_ctx40k_r1.md`](../../../../results/formal/agentdojo/RTXPro6000/agentdojo/task_injection_ctx40k_r1/qwen3_8b_task_injection_ctx40k_r1.md)

## 3. 原始数据

原始 AgentDojo 全量运行包含 4,745 条 trace：

- 有效 trace：4,709；
- 攻击成功：613；
- 攻击未成功：4,096；
- 正例比例：13.02%。

原始统计见：

[`trace_statistics.md`](../../../../results/formal/agentdojo/RTXPro6000/agentdojo/task_injection_ctx40k_r1/analysis/trace_statistics.md)

Qwen3-4B 特征提取最终得到：

- 已缓存特征：4,704 条；
- `rejected_or_failed`：5 条；
- 其中最开始的 8 条来自 smoke test，正式续跑时显示为 `skipped=8`；
- 正式续跑新增 `completed=4696`，因此总缓存数为 `8 + 4696 = 4704`。

被拒绝样本的具体原因以远端 `rejected.jsonl` 为准，不应静默截断后混入训练集。

## 4. 模型实际看到什么

输入文本由 `raw_trace.json` 的 `trajectory.messages` 序列化得到，包括：

- system message；
- user task；
- assistant message；
- tool call；
- tool result；
- final answer。

以下字段只作为划分和评估 metadata，不进入 Qwen3-4B 或 MLP 输入：

- `suite_name`；
- `attack_type`；
- `user_task_id`；
- `injection_task_id`；
- `group_id`；
- `ground_truth`；
- 文件路径和 sample ID。

虽然没有显式输入 suite 和 attack 标签，任务文本、工具名称和行为模式仍可能暴露 suite 或 attack 信息。因此“不显式输入 metadata”不等于模型无法学习领域先验。

原始 Qwen3-8B 运行可能开启了 thinking，但现有 trace 没有保存完整的 `reasoning_content`。本实验重新编码的是可见行为轨迹，不是在恢复 Qwen3-8B 生成时的内部 hidden state。

## 5. Qwen3-4B 特征提取方法

特征提取使用 Hugging Face Transformers `AutoModel`，不是 vLLM：

- `model.eval()`；
- `torch.inference_mode()`；
- `use_cache=False`；
- batch size 为 1；
- 不调用 `generate()`；
- temperature、top-p、top-k 和 thinking 配置不参与特征提取；
- forward hook 只把需要的位置和 pooling 移到 CPU。

长期特征库保存每个 decoder block 输出及最终 LayerNorm 输出。全局特征包括：

- `trace_last`：原始 trace 最后一个 token；
- `trace_last32_mean`：最后 32 个 token 的均值；
- `trace_mean`：完整 trace 的 token 均值；
- `readout_last`：追加固定安全判定问题后的最后 token。

还保存以下步骤边界的 hidden state：

- `tool_call_end`；
- `assistant_decision_end`；
- `tool_result_end`；
- `tool_round_end`。

实验 01 只使用：

```text
feature = trace_last
layer = -1
```

其中 `layer=-1` 表示模型最终 LayerNorm 的输出，不是倒数第一个 decoder block 的未归一化输出。

## 6. 数据划分

划分键为：

```text
(suite_name, user_task_id, injection_task_id)
```

同一 user task 与 injection task 组合下的所有攻击方式必须进入同一个 split，从而避免同一任务组合同时出现在训练集和测试集。

候选划分尽量匹配：

```text
suite_name x attack_type x label
```

的总体分布，但这些属性不进入模型输入。划分比例为：

- train：70%；
- validation：15%；
- test：15%。

当前测试集包含 705 条：

- 正例 89 条；
- 负例 616 条；
- 正例比例 12.62%。

这是固定的一次 group split。10 个训练随机种子共享相同 manifest，不能根据测试集结果挑选“最好 seed”。

## 7. 第一版分类头和训练配置

MLP 架构为：

```text
hidden state
  -> LayerNorm(input_dim)
  -> Linear(input_dim, 256)
  -> SiLU
  -> Dropout(0.1)
  -> Linear(256, 2)
  -> two-class logits
```

训练配置：

| 参数 | 值 |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-2` |
| Batch size | `128` |
| 最大 epoch | `50` |
| Early-stop patience | `5` |
| Loss | class-weighted cross entropy |
| 正类权重 | `negative_count / positive_count` |
| Batch sampling | 自然随机 batch，不强制 1:1 |

每个 epoch 在 validation 上计算 AP。最佳 checkpoint 按 **validation AP 最大** 保存。

在该 epoch 的 validation 分数上另外选择使 **F2 最大** 的阈值，并与 checkpoint 一起保存。F2 更偏重 Recall，因此当前决策策略倾向于少漏报、允许更多误报。

## 8. 远端运行环境

本实验使用与 NVIDIA 570.124.04 驱动兼容的 CUDA 12.8 PyTorch 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd-guard-features-cu128
```

原来的 `sd4-vllm-cu130` 环境安装了 `torch 2.13.0+cu130`，高于当前驱动可支持的 CUDA 12.8，因而 `torch.cuda.is_available()` 为 false。特征提取和分类头训练使用新的 cu128 环境。

运行前检查：

```bash
nvidia-smi

python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`flash-attn` 不是必需依赖。下面使用 PyTorch SDPA，避免额外安装和编译 `flash-attn`。

## 9. 公共环境变量

从 `Code/FormalExp` 目录运行所有模块：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd-guard-features-cu128

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export FORMAL_EXP=/root/autodl-tmp/Code/SD_Guard/Code/FormalExp
export TRACE_ROOT=/root/autodl-tmp/Code/SD_Guard/results/agentdojo/task_injection_ctx40k_r1/traces
export FEATURE_ROOT=/root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1
export QWEN4B=/root/autodl-tmp/models/Qwen3-4B
export OFFLINE_DATASET="$FEATURE_ROOT/offline_datasets/trace_last_layer_final"
export REPEATED_ROOT="$FEATURE_ROOT/experiments/trace_last_final_mlp_10seeds"

cd "$FORMAL_EXP"
```

说明：设置 `HF_ENDPOINT` 是为了记录 Mirror 地址；开启 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 后，正式运行只读取本地模型，Mirror 不会被访问。

## 10. 完整运行命令

### 10.1 Smoke test

```bash
python -m guard_training.qwen3_trace_features.extract_features \
  --trace-root "$TRACE_ROOT" \
  --output-dir "$FEATURE_ROOT" \
  --model-path "$QWEN4B" \
  --model-dtype bfloat16 \
  --storage-dtype float16 \
  --attn-implementation sdpa \
  --shard-size 128 \
  --continue-on-error \
  --limit 8
```

确认 smoke test 正常后，可以在同一输出目录续跑全量。脚本根据 sample ID 跳过已经完成的 8 条。

### 10.2 全量提取 Qwen3-4B hidden state

```bash
python -m guard_training.qwen3_trace_features.extract_features \
  --trace-root "$TRACE_ROOT" \
  --output-dir "$FEATURE_ROOT" \
  --model-path "$QWEN4B" \
  --model-dtype bfloat16 \
  --storage-dtype float16 \
  --attn-implementation sdpa \
  --shard-size 128 \
  --continue-on-error
```

以 `$FEATURE_ROOT/feature_config.json` 记录的 `attn_implementation`、模型 hash、tokenizer 和层数为本次实际运行的最终事实来源。

### 10.3 验证特征分片

```bash
python -m guard_training.qwen3_trace_features.inspect_features \
  --feature-root "$FEATURE_ROOT" \
  --check-finite \
  --output "$FEATURE_ROOT/feature_report.json"
```

辅助计数：

```bash
find "$FEATURE_ROOT/features" -name 'shard_*.jsonl' -print0 \
  | xargs -0 cat \
  | wc -l

wc -l "$FEATURE_ROOT/rejected.jsonl"
```

本次结果应为：

```text
feature metadata rows: 4704
rejected rows: 5
```

### 10.4 构建固定 group split

```bash
python -m guard_training.frozen_hidden_mlp.build_split_manifest \
  --feature-root "$FEATURE_ROOT" \
  --output "$FEATURE_ROOT/split_manifest.jsonl" \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 20261106 \
  --trials 10000
```

同时生成：

```text
$FEATURE_ROOT/split_manifest.summary.json
```

### 10.5 物化紧凑离线数据集

13GB 左右的长期特征库保留所有层、pooling 和工具步骤。实验 01 只导出最终层 `trace_last`，避免每次训练都扫描总特征库：

```bash
python -m guard_training.frozen_hidden_mlp.materialize_dataset \
  --feature-root "$FEATURE_ROOT" \
  --manifest "$FEATURE_ROOT/split_manifest.jsonl" \
  --output-dir "$OFFLINE_DATASET" \
  --feature trace_last \
  --layer -1 \
  --storage-dtype float16
```

输出：

```text
dataset_config.json
train.safetensors
train.metadata.jsonl
validation.safetensors
validation.metadata.jsonl
test.safetensors
test.metadata.jsonl
```

### 10.6 单个 seed 训练

```bash
python -m guard_training.frozen_hidden_mlp.train \
  --offline-dataset "$OFFLINE_DATASET" \
  --output-dir "$FEATURE_ROOT/experiments/trace_last_final_mlp_seed42" \
  --head mlp \
  --hidden-dim 256 \
  --dropout 0.1 \
  --positive-weight inverse \
  --learning-rate 1e-3 \
  --weight-decay 1e-2 \
  --batch-size 128 \
  --epochs 50 \
  --patience 5 \
  --seed 42
```

独立测试：

```bash
python -m guard_training.frozen_hidden_mlp.evaluate \
  --offline-dataset "$OFFLINE_DATASET" \
  --checkpoint "$FEATURE_ROOT/experiments/trace_last_final_mlp_seed42/best.pt" \
  --output-dir "$FEATURE_ROOT/experiments/trace_last_final_mlp_seed42/evaluation" \
  --split test
```

### 10.7 运行 10 个随机种子并聚合

正式实验使用：

```bash
python -m guard_training.frozen_hidden_mlp.run_repeated \
  --offline-dataset "$OFFLINE_DATASET" \
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
  --patience 5
```

脚本依次完成每个 seed 的训练、测试和最终聚合。若 checkpoint 和测试结果已经存在，默认跳过；只有明确需要覆盖时才使用 `--force`。

## 11. 输出文件

每个 `seed_N/` 保存：

```text
best.pt
train_config.json
history.jsonl
best_validation_metrics.json
train.log
evaluate.log
evaluation/test_metrics.json
evaluation/test_predictions.jsonl
```

10 次实验总目录保存：

```text
aggregate_test_metrics.json
aggregate.log
```

`best.pt` 包含分类头权重、特征类型、层号、模型结构、训练配置、validation 阈值和最佳 validation 指标。

## 12. 复现实验时必须保持固定的项目

与实验 01 对比的新实验应尽量保持以下项目不变：

- 同一个 `split_manifest.jsonl`；
- 同一个 test 集；
- 同一组 seeds `42..51`；
- 相同训练 epoch、patience 和 batch size；
- checkpoint 仍只由 validation 指标选择；
- 阈值仍只在 validation 上选择；
- 不根据 test 结果挑选 seed 或超参数。

只改变当前需要研究的变量，例如 pooling、layer、分类头或正类权重。

