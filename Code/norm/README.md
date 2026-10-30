# Qwen3Guard-Stream thinking 对比实验

当前实验只使用 `Qwen3GuardTest` 的 `thinking` split。默认使用固定种子从完整
split 中随机抽取 100 条样本，并让以下三种方法复用完全相同的样本清单：

1. 自己实现的逐 token `stream_moderate_from_ids`；
2. 自己实现的一次性完整 causal forward；
3. 官方 `eval_stream.py` 的一次性完整 causal forward。

三种方法都报告 strict 和 loose：

- strict：存在连续的 `Unsafe, Unsafe` 或 `Controversial, Controversial`；
- loose：只在存在连续的 `Unsafe, Unsafe` 时判为 Unsafe。

Streaming 的 strict/loose 按 token 顺序即时触发。Streaming strict 选择最早出现
的同类非 Safe 标签对；官方 strict 会先全局查找 `Unsafe, Unsafe`，不存在时再
查找 `Controversial, Controversial`。两者使用相同 token 标签序列时最终二分类
一致，但触发位置可能不同。

## 文件

```text
norm/
├── eval_stream_guard.py            # Streaming + 自己的完整 forward
├── analyze_stream_results.py       # 三种方法的指标与一致性分析
├── official/
│   ├── eval_stream.py              # 未改动的官方脚本
│   └── eval_stream_sampled.py      # 官方逻辑 + manifest/单卡包装
├── guard_eval/                     # 公共实现
├── scripts/
│   └── run_thinking_comparison.sh  # 顺序执行三种方法
└── tests/
    └── test_pure_logic.py
```

旧的 loc、数据检查和多模型批量脚本不再属于正式实验流程。

## 单卡约束

每个模型必须完整放在一张 GPU。运行命令通过 `CUDA_VISIBLE_DEVICES` 只暴露一张
物理卡，自定义评测再使用：

```python
device_map={"": 0}
```

加载完成后会检查所有参数、buffer 和 Hugging Face device map。发现跨卡、CPU
offload 或 disk offload 时立即停止。

`CUDA_VISIBLE_DEVICES=1` 选择物理 GPU 1 后，Python 进程内部仍将它显示为逻辑
`cuda:0`，这是正常现象。

## 定位 split 的 full-forward 评测

`eval_guard_location_full_forward.py` 专门评测 `thinking_loc` 和
`response_loc`。它使用与官方脚本一致的单次 causal full forward 得到每个
Assistant token 的风险标签，再按时间顺序重建两种在线策略：

- strict：最早出现的同类非 Safe 相邻对，即 `UU` 或 `CC`；
- loose：最早出现的 `UU`。

触发位置取相邻对的第二个 token，并使用完整模板序列中的零基索引，与数据集的
闭区间 `[unsafe_start_index, unsafe_end_index]` 比较。定位评测强制要求重新分词的
`input_ids` 与数据集标注完全一致。

完整运行两个定位 split：

```bash
cd /root/autodl-tmp/SD_Guard/Code/norm

CUDA_VISIBLE_DEVICES=0 python -u eval_guard_location_full_forward.py \
  --model-path /root/autodl-tmp/SD_Guard/models/Qwen3Guard-Stream-4B \
  --dataset-path /root/autodl-tmp/SD_Guard/datasets/Qwen3GuardTest \
  --output-dir outputs/location-full-forward-4B \
  --overwrite
```

先各跑前3条进行冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_guard_location_full_forward.py \
  --model-path /root/autodl-tmp/SD_Guard/models/Qwen3Guard-Stream-4B \
  --dataset-path /root/autodl-tmp/SD_Guard/datasets/Qwen3GuardTest \
  --output-dir outputs/location-smoke-4B \
  --max-samples 3 \
  --overwrite
```

输出包括：

```text
outputs/location-full-forward-4B/
├── run_metadata.json
├── summary.json
├── metrics.csv
└── raw/
    ├── thinking_loc/
    │   ├── token_records.jsonl
    │   ├── sample_records.jsonl
    │   └── summary.json
    └── response_loc/
        ├── token_records.jsonl
        ├── sample_records.jsonl
        └── summary.json
```

每个 split、每个策略分别报告 `before / within / after / missed`、完整回答
`response_fnr`、截至人工不安全区间末尾仍未检出的 `timely_failure_rate`、
`exact_hit_rate`、`first_128_rate` 和官方风格位置分桶。该脚本不测真实 KV-cache
streaming 延迟；需要验证 full-forward 重建与真实 streaming 的一致性时，继续使用
`eval_stream_guard.py` 抽样对照。

## 完整 thinking split 的统一 full-forward 评测

`eval_guard_thinking_full_forward.py` 只运行完整的 `thinking` split。模型对每条
response 做一次 causal full forward，再从逐 token 标签按时间顺序重建 strict 和
loose 决策：

- 全部 Safe/Unsafe 样本计算 TP、TN、FP、FN、FPR、FNR、Precision、Recall、F1、
  Accuracy；
- Unsafe 样本使用自带的 `unsafe_start_index` / `unsafe_end_index`，无缝复用上述
  `before / within / after / missed` 和及时性指标；
- Safe 样本统计首次误触发位于 response 的第几个 token、回答进度比例、UU/CC
  来源，以及前 32/64/128 token 的累计误报率。

完整运行：

```bash
cd /root/autodl-tmp/SD_Guard/Code/norm

CUDA_VISIBLE_DEVICES=0 python -u eval_guard_thinking_full_forward.py \
  --model-path /root/autodl-tmp/SD_Guard/models/Qwen3Guard-Stream-4B \
  --dataset-path /root/autodl-tmp/SD_Guard/datasets/Qwen3GuardTest \
  --output-dir /root/autodl-tmp/SD_Guard/outputs/thinking-full-forward-4B \
  --overwrite
```

先运行前 10 条做冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_guard_thinking_full_forward.py \
  --model-path /root/autodl-tmp/SD_Guard/models/Qwen3Guard-Stream-4B \
  --dataset-path /root/autodl-tmp/SD_Guard/datasets/Qwen3GuardTest \
  --output-dir /root/autodl-tmp/SD_Guard/outputs/thinking-full-forward-smoke-4B \
  --max-samples 10 \
  --overwrite
```

主要输出：

```text
thinking-full-forward-4B/
├── run_metadata.json
├── summary.json
├── metrics.csv                       # strict/loose 统一主表
├── classification_metrics.csv        # 完整二分类指标
├── unsafe_location_metrics.csv       # Unsafe 区间及时性
├── safe_false_positive_metrics.csv   # Safe 误报位置
└── raw/thinking/
    ├── token_records.jsonl
    ├── sample_records.jsonl
    ├── false_positive_records.jsonl
    ├── false_negative_records.jsonl
    └── late_or_missed_unsafe_records.jsonl
```

其中 Unsafe 的 `before` 是提前预警，不属于误报；FPR 只使用 ground-truth Safe
样本作为分母。这个入口只做 full forward，不测真实 KV-cache streaming 延迟。

## 一条命令完成对比

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/norm

GUARD_GPU=0 bash scripts/run_thinking_comparison.sh \
  /home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest \
  /home/model_and_data/yxz/model/Qwen3Guard-Stream-4B \
  outputs/compare-4B-100 \
  100 \
  42
```

三个进程按顺序运行，不会同时争用 GPU：

1. 自定义 Streaming 与完整 forward；
2. 官方完整 forward；
3. 不加载模型的离线分析。

可通过环境变量指定远程 Python：

```bash
PYTHON_BIN=/home/model_and_data/yxz/envs/SD_Guard/bin/python \
GUARD_GPU=0 \
bash scripts/run_thinking_comparison.sh DATASET_PATH MODEL_PATH OUTPUT_DIR 100 42
```

## 分步运行

### 1. Streaming 与自己的完整 forward

```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_stream_guard.py \
  --model-path /home/model_and_data/yxz/model/Qwen3Guard-Stream-4B \
  --dataset-path /home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest \
  --output-dir outputs/compare-4B-100/custom \
  --sample-count 100 \
  --seed 42 \
  --overwrite
```

它生成 `sample_manifest.json`。manifest 同时保存随机种子、dataset index、sample ID
和执行顺序。不要为官方评测重新抽样。

Streaming 每个 token 的计时范围是：

```text
Assistant token ID 已准备好
→ CUDA synchronize
→ stream_moderate_from_ids
→ CUDA synchronize
→ 提取 risk_level/risk_prob
→ 计时结束
```

不包含 Assistant 生成、Prompt prefill、模板与分词、完整 forward、pair 策略更新和
结果写盘。即使回答已经触发 Unsafe，仍会处理后续 token，以保存完整延迟分布。

### 2. 官方逻辑，同一批样本

```bash
CUDA_VISIBLE_DEVICES=0 python -u official/eval_stream_sampled.py \
  --model-path /home/model_and_data/yxz/model/Qwen3Guard-Stream-4B \
  --dataset-path /home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest \
  --sample-manifest outputs/compare-4B-100/custom/sample_manifest.json \
  --output-path outputs/compare-4B-100/official_predictions.jsonl \
  --overwrite
```

包装脚本直接导入旁边未改动官方脚本的 `ContentEvaluator`。它只增加 manifest
筛选、单卡检查和元数据，不修改官方 chat template、完整 forward 或风险头解码。

### 3. 离线分析

```bash
python -u analyze_stream_results.py \
  --custom-dir outputs/compare-4B-100/custom \
  --official-output outputs/compare-4B-100/official_predictions.jsonl \
  --output-dir outputs/compare-4B-100/analysis \
  --overwrite
```

官方脚本原始 `evaluate_f1` 把 strict 和 loose 的 prediction 列表传反。分析脚本不
读取这两个打印名称，而是从官方保存的 `pred_risk_levels` 重新计算正确指标。

## 指标

三种方法的 strict/loose 分别报告：

- TP、TN、FP、FN；
- FPR、FNR；
- Precision、Recall、F1；
- Accuracy。

一致性报告包括：

- Streaming 与自己的完整 forward 的 token 标签一致率；
- Streaming 与自己的完整 forward 的样本预测一致率；
- 自己的完整 forward 与官方脚本的 input IDs、Response 边界和 token 标签一致率；
- strict/loose 的不一致样本 ID。

Streaming 延迟报告：

- Mean、P50、P95、P99、Min、Max；
- 总 Assistant token 数；
- Guard token/s；
- 每条回答的 `response_token_count`、`mean_context_length` 和
  `mean_token_latency_ms`；
- 平均上下文长度与平均 token 延迟的 Spearman 相关系数。

不再生成 loc 指标、长度四分位表或散点图。

## 输出

```text
outputs/compare-4B-100/
├── custom/
│   ├── sample_manifest.json
│   └── raw/
│       ├── run_metadata.json
│       ├── token_records.jsonl
│       ├── sample_records.jsonl
│       ├── full_forward_records.jsonl
│       └── mismatch_records.jsonl
├── official_predictions.jsonl
├── official_predictions.jsonl.metadata.json
└── analysis/
    ├── summary.json
    ├── metrics.csv
    ├── latency_summary.csv
    ├── sample_latency.csv
    └── disagreements.jsonl
```

## 复用同一批样本测试其他模型

把第一次运行生成的 manifest 传给另一个模型：

```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_stream_guard.py \
  --model-path /home/model_and_data/yxz/model/Qwen3Guard-Stream-0.6B \
  --dataset-path /home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest \
  --output-dir outputs/compare-0.6B-same-100/custom \
  --sample-manifest outputs/compare-4B-100/custom/sample_manifest.json \
  --overwrite
```

## 纯逻辑测试

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
