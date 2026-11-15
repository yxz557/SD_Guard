# Readout-conditioned Tool-step Attention

本模块实现最终层隐藏状态上的动态attention实验：

```text
c_qattn             readout查询全部tool_call_end，分类器只接收call_attn
t_qattn             readout查询全部tool_result_end，分类器只接收result_attn
rc_qattn            [readout, call_attn]
rt_qattn            [readout, result_attn]
rct_independent     decision/result分别使用独立动态attention
rct_shared          decision计算权重，同一round的result共享该权重
```

所有方法都只使用`layer=-1`（最终LayerNorm输出）。call/result不会混合mean。共享方法使用`assistant_decision_end ↔ tool_round_end`按round配对；未返回result的最后一轮保留decision，并用mask排除缺失result。

## 0. 先运行Mean模型参数量消融

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export MEAN_DATASETS="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/offline_datasets/tool_step_mean_layer_final"
export PARAM_RUNS="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/experiments/tool_step_mean_parameter_ablation"

python -m guard_training.tool_step_mean_pool.run_parameter_ablation \
  --dataset-root "$MEAN_DATASETS" \
  --output-root "$PARAM_RUNS" \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --positive-weight none \
  --dropout 0.1 \
  --learning-rate 1e-3 \
  --weight-decay 1e-2 \
  --batch-size 128 \
  --epochs 50 \
  --patience 5 \
  --device auto
```

该命令运行：

```text
c0_h256 / t0_h256
rc_h256 / rt_h256 / rct_h256
rc_pm_h127 / rt_pm_h127 / rct_pm_h84
c0_large_h514 / t0_large_h514 / c0_large_h771
```

结果在`$PARAM_RUNS/comparison/`。

## 1. 一次性构建变长Attention离线数据

该步骤扫描13GB完整特征库一次。后续训练只读取最终层变长离线数据，不再读取13GB。

```bash
export BASE_OFFLINE="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/offline_datasets/readout_last_layer_final"
export LOCAL_FEATURE_ROOT="$SD_ROOT/results/formal/guard_training/qwen3_4b_task_injection_nothink_ctx40k_r1"
export PUBLIC_FEATURE_ROOT="$SD_ROOT/results/formal/guard_training/qwen3_4b_agentdojo_public_v1_2_1_no_defense"
export ATTN_DATASET="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/offline_datasets/tool_step_attention_layer_final"

python -m guard_training.tool_step_attention.build_offline_dataset \
  --input-dataset "$BASE_OFFLINE" \
  --feature-root "$LOCAL_FEATURE_ROOT" \
  --feature-root "$PUBLIC_FEATURE_ROOT" \
  --output-dir "$ATTN_DATASET" \
  --layer -1 \
  --storage-dtype float16
```

先检查：

```bash
cat "$ATTN_DATASET/build_report.json"
cat "$ATTN_DATASET/dataset_config.json"
wc -l "$ATTN_DATASET"/*.metadata.jsonl
```

## 2. 运行全部Attention实验

```bash
export ATTN_RUNS="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/experiments/tool_step_qattn_none_10seeds"

python -m guard_training.tool_step_attention.run_experiments \
  --offline-dataset "$ATTN_DATASET" \
  --output-root "$ATTN_RUNS" \
  --variants c_qattn t_qattn rc_qattn rt_qattn rct_independent rct_shared \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --attention-dim 128 \
  --head-hidden-dim 256 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --positive-weight none \
  --learning-rate 1e-3 \
  --weight-decay 1e-2 \
  --batch-size 64 \
  --epochs 50 \
  --patience 5 \
  --device auto
```

如果原mean实验仍在服务器，可在同一对比报告中加入基线：

```bash
export MEAN_RUNS="$SD_ROOT/results/formal/guard_training/nothink_public_v1_2_1/experiments/tool_step_mean_none_10seeds"

# 在上一条run_experiments命令中追加：
  --reference C0="$MEAN_RUNS/c0" \
  --reference T0="$MEAN_RUNS/t0" \
  --reference RC-Mean="$MEAN_RUNS/rc" \
  --reference RT-Mean="$MEAN_RUNS/rt" \
  --reference RCT-Mean="$MEAN_RUNS/rct"
```

每个seed保存checkpoint、loss history、F2阈值test指标以及validation上选择的固定工作点：

```text
FPR <= 5%
FPR <= 10%
Recall >= 95%
```

`evaluation/test_predictions.jsonl`还会保存每条trace权重最高的call/result/round，便于检查attention是否关注真正危险步骤。

汇总结果位于：

```text
$ATTN_RUNS/comparison/comparison.md
$ATTN_RUNS/comparison/comparison.json
$ATTN_RUNS/comparison/overall_metrics.csv
$ATTN_RUNS/comparison/overall_metrics.png
```

## 3. 建议执行顺序

```text
1. 参数量消融
2. 构建Attention离线数据
3. 先用seeds 42 43 44做三次冒烟实验
4. 确认loss与round配对正常后补齐10 seeds
```

三seed预实验只需把`--seeds`改为：

```bash
--seeds 42 43 44
```

