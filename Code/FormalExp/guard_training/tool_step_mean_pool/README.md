# Tool-step mean pooling：C0 / T0 / RC / RT / RCT

本模块复用已经提取完成的 Qwen3-4B hidden-state 特征，不重新运行 backbone prefill。

## 实验定义

对每条 trace 的最终层逐步特征，分别计算：

```text
call_pool   = mean(all tool_call_end)
result_pool = mean(all tool_result_end)

C0  = call_pool
T0  = result_pool
RC  = concat(readout_last, call_pool)
RT  = concat(readout_last, result_pool)
RCT = concat(readout_last, call_pool, result_pool)
```

call 与 result 只在各自类别内部做 mean pooling，二者不会混在一起平均。

首轮实验只保留同时满足 `n_tool_calls > 0 AND n_tool_results > 0` 的样本。根据已有侦察结果，主要排除的是 20 条既没有 call 也没有 result 的反例；`call > result`、工具返回错误、攻击执行成功但 user task 未完成等样本都保留。

为了严格比较，推荐额外生成 `R0-common = readout_last`。它不是第六种新方法，而是经过同一过滤条件的 readout 基线，避免旧 R0 比五种新实验多 20 条样本。

## 1. 构建六套紧凑离线数据（五种新实验 + 公平基线）

在服务器的 `Code/FormalExp` 目录执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd-guard-features-cu128

cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export BASE_OFFLINE="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/offline_datasets/readout_last_layer_final"
export LOCAL_FEATURE_ROOT="$SD_ROOT/results/formal/guard_training/qwen3_4b_task_injection_nothink_ctx40k_r1"
export PUBLIC_FEATURE_ROOT="$SD_ROOT/results/formal/guard_training/qwen3_4b_agentdojo_public_v1_2_1_no_defense"
export TOOL_DATASETS="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/offline_datasets/tool_step_mean_layer_final"

python -m guard_training.tool_step_mean_pool.build_offline_datasets \
  --input-dataset "$BASE_OFFLINE" \
  --feature-root "$LOCAL_FEATURE_ROOT" \
  --feature-root "$PUBLIC_FEATURE_ROOT" \
  --output-root "$TOOL_DATASETS" \
  --layer -1 \
  --storage-dtype float16 \
  --include-r0-common
```

输出：

```text
tool_step_mean_layer_final/
├── build_report.json
├── excluded_by_eligibility.jsonl
├── r0_common/
├── c0/
├── t0/
├── rc/
├── rt/
└── rct/
```

`build_report.json` 记录每个 split 过滤前后样本量、正负例、source、suite、attack 和 group 分布。脚本拒绝覆盖已有输出目录；如果需要重建，请换一个新目录或人工确认后处理旧目录。

## 2. 一次运行全部实验并自动对比

以下设置与已确定的首轮方案一致：MLP hidden dim 256、无 class weight、10 个 seeds、最多 50 epochs、patience 5。

```bash
export TOOL_RUNS="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/experiments/tool_step_mean_none_10seeds"

python -m guard_training.tool_step_mean_pool.run_experiments \
  --dataset-root "$TOOL_DATASETS" \
  --output-root "$TOOL_RUNS" \
  --include-r0-common \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --head mlp \
  --hidden-dim 256 \
  --dropout 0.1 \
  --positive-weight none \
  --learning-rate 1e-3 \
  --weight-decay 1e-2 \
  --batch-size 128 \
  --epochs 50 \
  --patience 5 \
  --device auto
```

该命令依次训练 `R0-common/C0/T0/RC/RT/RCT`，每种都调用现有 `run_repeated`，因此每个 seed 仍保存独立 checkpoint、validation 阈值、history、test 指标和预测。每种实验还会保存 `training_validation_loss.png`。

最终对比位于：

```text
$TOOL_RUNS/comparison/comparison.json
$TOOL_RUNS/comparison/comparison.md
$TOOL_RUNS/comparison/overall_metrics.csv
$TOOL_RUNS/comparison/overall_metrics.png
```

对比包含总体及按 suite、attack 拆分的 AP、ROC-AUC、accuracy、precision、recall、FPR、FNR，并检查各实验 test sample ID 是否完全相同。

## 3. 只跑五种新方法

如果不想重跑公平版 R0，构建和训练时都去掉 `--include-r0-common`。但旧 R0 含有额外的无工具样本，和新实验不是严格同一 test population；这种对比应在报告中注明。

## 4. 单独重做对比报告

```bash
python -m guard_training.tool_step_mean_pool.compare_experiments \
  --experiment R0-common="$TOOL_RUNS/r0_common" \
  --experiment C0="$TOOL_RUNS/c0" \
  --experiment T0="$TOOL_RUNS/t0" \
  --experiment RC="$TOOL_RUNS/rc" \
  --experiment RT="$TOOL_RUNS/rt" \
  --experiment RCT="$TOOL_RUNS/rct" \
  --output-dir "$TOOL_RUNS/comparison"
```

## 解释限制

- mean pooling 是最简单的第一版，会把全部正常步骤与少数危险步骤平均，可能产生危险信号稀释；它的价值是建立 attention pooling 之前的清晰基线。
- `RC/RT/RCT` 使用拼接后的更高维输入，但 MLP hidden dim 固定为 256，因此输入层参数量会随维度增加。报告会记录各实验参数量；若组合方法胜出，后续仍需做“等参数量投影”消融，区分信息增益与容量增益。
- 标签仍是整条 trace 的攻击是否成功，不是逐步骤危险标签，所以本实验判断的是哪些步骤集合有助于 trace-level 分类。
