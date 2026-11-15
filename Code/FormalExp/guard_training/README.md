# SD_Guard：完整 Trace 冻结特征分类器

## 实验文档

- [实验 01：方法、配置与完整运行命令](docs/experiment_01_trace_last_mlp_method.md)
- [实验 01：10-seed 结果与分组分析](docs/experiment_01_trace_last_mlp_results.md)

该目录实现第一阶段 guard 训练流程：

1. 使用普通 `Qwen/Qwen3-4B` 对每条完整 AgentDojo trace 做一次纯 forward/prefill；
2. 保存多个层、多个 pooling 以及每个工具步骤边界的 hidden state；
3. 按任务组合分组构建无泄漏的 train/validation/test manifest；
4. 离线训练 Linear/MLP 分类头；
5. 在固定 test 集上评估，并可导出每一步的探索性风险曲线。

## 目录职责

```text
guard_training/
├── common/                  # trace、序列化、特征分片、指标和划分工具
├── qwen3_trace_features/    # Qwen3-4B 特征提取
├── frozen_hidden_mlp/       # 划分、训练、测试、步骤风险导出
├── tool_step_mean_pool/     # call/result mean pooling 与组合消融
└── tests/                   # 不依赖模型权重的单元测试
```

工具调用/返回步骤的 C0、T0、RC、RT、RCT 实验及完整运行命令见
[tool_step_mean_pool/README.md](tool_step_mean_pool/README.md)。

大体积特征应写入 `results/formal/guard_training/`，不要提交到源码目录。

## 输入约束

模型文本只由 `raw_trace.json` 的 `trajectory.messages` 构成，包括 system、user、assistant、tool call、tool result 和 final answer。

以下字段只保留在 metadata，不进入模型输入：

- `suite_name`
- `attack_type`
- `user_task_id`
- `injection_task_id`
- `group_id`
- `ground_truth`
- 文件路径和 sample ID

标签来自相邻 `ground_truth.json` 的 `hijacked`。无效 trace 会被排除。

现有 r1 trace 没有保存 Qwen3-8B 的显式 reasoning，因此这里提取的是 system/user/assistant/tool 行为轨迹的表示，不包含已经丢失的 thinking 文本。Qwen3-4B 会用确定性的统一格式重新编码这些消息；这不是恢复原生成模型当时的内部 hidden state。

特征提取只有 forward，不调用 `generate()`，所以 temperature、top-p、top-k 和 `enable_thinking` 不参与这一步。它们只影响原 trace 是如何生成的。

## 保存的特征

每个固定长度特征的形状为 `[samples, saved_layers, hidden_size]`：

- `trace_last`：原始 trace 最后一个 token；
- `trace_last32_mean`：原始 trace 最后 32 个 token 的平均；
- `trace_mean`：原始 trace 全部 token 的平均；
- `readout_last`：在 trace 后追加固定安全判定问题后的最后 token。

`saved_layers` 包括所有 decoder block 输出，以及模型最终 LayerNorm 输出。`--layer -1` 表示最终 LayerNorm 输出。

工具步骤保存在 `step_hidden`，形状为 `[all_steps, saved_layers, hidden_size]`，对应：

- `tool_call_end`
- `assistant_decision_end`
- `tool_result_end`
- `tool_round_end`

同一次 causal forward 中，早期边界不受后续 token 影响，因此这些向量等价于对对应前缀单独执行 prefill。

## 1. 提取 Qwen3-4B 特征

从 `Code/FormalExp` 目录运行模块，以下为远端 Linux 示例：

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m guard_training.qwen3_trace_features.extract_features \
  --trace-root /root/autodl-tmp/Code/SD_Guard/results/agentdojo/task_injection_ctx40k_r1/traces \
  --output-dir /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1 \
  --model-path /root/autodl-tmp/models/Qwen3-4B \
  --model-dtype bfloat16 \
  --storage-dtype float16 \
  --attn-implementation flash_attention_2 \
  --shard-size 128 \
  --continue-on-error
```

建议先冒烟测试：

```bash
python -m guard_training.qwen3_trace_features.extract_features \
  ... \
  --limit 8
```

脚本使用 forward hooks，只把所需位置或 pooling 转移到 CPU，不启用 `output_hidden_states=True`，也不生成新 token。提取使用 `torch.inference_mode()`、`use_cache=False` 和 batch size 1。

超过 `--max-length` 或模型 `max_position_embeddings` 的样本不会静默截断，而会写入 `rejected.jsonl`。

提取完成后先验证所有分片：

```bash
python -m guard_training.qwen3_trace_features.inspect_features \
  --feature-root /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1 \
  --check-finite \
  --output /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1/feature_report.json
```

## 2. 构建固定数据划分

```bash
python -m guard_training.frozen_hidden_mlp.build_split_manifest \
  --feature-root /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1 \
  --output /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1/split_manifest.jsonl \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 20261106 \
  --trials 10000
```

分组键固定为：

```text
(suite_name, user_task_id, injection_task_id)
```

同一任务组合的所有攻击方式必须进入同一个 split。候选划分会尽量匹配 `suite × attack_type × label` 的总体分布，但这些元数据不会输入模型。

## 3. 物化第一版紧凑离线数据集

13GB左右的总特征库用于长期保留所有层、pooling 和工具步骤。第一版 MLP 只需要 `trace_last` 的 `layer=-1`，应先一次性导出紧凑 train/validation/test 文件：

```bash
export OFFLINE_DATASET="$FEATURE_ROOT/offline_datasets/trace_last_layer_final"

python -m guard_training.frozen_hidden_mlp.materialize_dataset \
  --feature-root "$FEATURE_ROOT" \
  --manifest "$FEATURE_ROOT/split_manifest.jsonl" \
  --output-dir "$OFFLINE_DATASET" \
  --feature trace_last \
  --layer -1 \
  --storage-dtype float16
```

输出为：

```text
dataset_config.json
train.safetensors
train.metadata.jsonl
validation.safetensors
validation.metadata.jsonl
test.safetensors
test.metadata.jsonl
```

对约4700条、数千维 hidden state，三份紧凑特征通常只有几十 MB。后续所有随机种子训练都只读取这些文件，不再扫描13GB总特征库。

## 4. 训练第一版 MLP

```bash
python -m guard_training.frozen_hidden_mlp.train \
  --offline-dataset "$OFFLINE_DATASET" \
  --output-dir /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1/experiments/trace_last_layer_last_mlp_seed42 \
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

`--positive-weight` 支持：

- `none`
- `sqrt_inverse`
- `inverse`
- 任意正浮点数，例如 `4.0` 或 `6.68`

第一版使用自然随机 batch 和 class-weighted cross entropy，不强制每个 batch 正负 1:1。

Linear head 基线只需改为：

```bash
--head linear
```

## 5. 独立测试

训练过程只查看 validation。固定最佳 checkpoint 和 validation 上选择的 F2 阈值后，再运行：

```bash
python -m guard_training.frozen_hidden_mlp.evaluate \
  --offline-dataset "$OFFLINE_DATASET" \
  --checkpoint /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1/experiments/trace_last_layer_last_mlp_seed42/best.pt \
  --output-dir /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1/experiments/trace_last_layer_last_mlp_seed42/evaluation \
  --split test
```

输出总体、按 suite 和按 attack 的指标与逐样本预测。

每次训练都会保存独立 `best.pt`，包含 MLP/Linear权重、输入维度、特征定义、validation阈值和最佳 validation 指标。

## 6. 连续运行10个随机种子并聚合

```bash
export REPEATED_ROOT="$FEATURE_ROOT/experiments/trace_last_final_mlp_10seeds"

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

每个 `seed_N/` 单独保存：

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

总目录额外保存：

```text
aggregate_test_metrics.json
aggregate.log
```

聚合报告包含每个数值指标的 mean、sample standard deviation、min、max 和10次原始值，也保留每次 run 的 checkpoint 和指标路径。10次实验必须共用同一个 split manifest，且不能根据 test 结果挑选“最好的一次”。

每个 `history.jsonl` 的每轮记录同时包含 `train_loss` 和
`validation_loss`。两者使用该次实验 `--positive-weight` 解析出的同一个
cross-entropy 目标。画出多个随机种子的个体曲线、均值与标准差带：

```bash
python -m guard_training.frozen_hidden_mlp.plot_loss_curves \
  --runs-root "$REPEATED_ROOT" \
  --output "$REPEATED_ROOT/training_validation_loss.png"
```

画图是独立步骤，不会因画图环境问题影响已经完成的训练。

### 6.1 从现有 offline dataset 构建 suite-only 子集

如果要在不改变原 task-pair split 的前提下训练 Travel-only 分类头，
直接过滤已经提取的 compact hidden-state dataset：

```bash
export TRAVEL_DATASET="$FEATURE_ROOT/offline_datasets/travel_only_trace_last_layer_final"

python -m guard_training.frozen_hidden_mlp.filter_offline_dataset \
  --input-dataset "$OFFLINE_DATASET" \
  --output-dir "$TRAVEL_DATASET" \
  --suite travel
```

该命令会筛选所有已存在的 split，包括辅助的
`public_validation` 和 `public_test`；它不重新切分 task pair，不需要重新运行
backbone prefill。输出目录必须尚不存在，以避免意外覆盖旧实验。

## 7. 导出工具步骤风险曲线

```bash
python -m guard_training.frozen_hidden_mlp.score_steps \
  --feature-root /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1 \
  --manifest /root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/qwen3_4b_task_injection_ctx40k_r1/split_manifest.jsonl \
  --checkpoint /path/to/best.pt \
  --output /path/to/test_step_scores.jsonl \
  --split test
```

重要限制：当前 head 使用整条 trace 标签训练。把它应用到中间步骤得到的是探索性 prefix-risk score，不是经过 step-level 标签监督的危险概率。不能把最终攻击成功标签复制给一条轨迹中的所有步骤。后续如果训练 step classifier，需要先从攻击目标的真实执行位置构造“首次危险步骤”标签。

## 推荐的首轮消融

保持 split manifest 不变，依次比较：

1. `trace_last`、`layer=-1`、Linear；
2. `trace_last`、`layer=-1`、MLP；
3. `trace_last32_mean`、`layer=-1`；
4. `trace_mean`、`layer=-1`；
5. `readout_last`、`layer=-1`；
6. `trace_last` 的倒数第 2/4/8 层；
7. positive weight 为 `none`、`sqrt_inverse`、`4.0`、`inverse`。

任何消融都只读取离线特征，不重新运行 Qwen3-4B。

