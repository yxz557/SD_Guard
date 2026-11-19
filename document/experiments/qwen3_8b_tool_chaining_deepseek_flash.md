# Qwen3-8B AgentLAB Tool Chaining 完整运行说明

## 1. 文档目的

本文记录 SD_Guard 正式实验中 AgentLAB Tool Chaining 的任务含义、运行脚本、
模型配置、vLLM 启动方式、smoke 测试、200 条全量实验、输出文件和完整性验收方法。

当前推荐方式是：使用更新后的 `Tool-chaining.py`，将混合数据集中的
`167` 条 SHADE_Arena 样本和 `33` 条 Agent_SafetyBench 样本在同一次正式运行中
全部处理，最终直接得到 `200` 条结果和 `200` 份完整 trace。

历史实验曾先得到 167 条 SHADE_Arena 结果，再补跑 33 条 Agent_SafetyBench 并
进行整合。该历史结果已整理成统一的 200 条数据，但以后从头复跑不需要再次拆成
两次。

## 2. Tool Chaining 是什么

Tool Chaining 测试攻击者是否能够通过多轮规划和工具调用，引导目标 Agent 完成
危险目标。每条样本包括：

- 用户/攻击任务描述；
- 攻击目标；
- 可用环境与工具定义；
- 经验证的参考工具链；
- planner 生成的阶段性提示；
- victim 模型的完整消息、工具调用和工具返回；
- judge 对危险目标完成度的评分。

当前混合数据集为：

```text
data/filtered_top_200_attacks.json
```

其中：

| 数据来源 | 样本数 | 环境实现 |
|---|---:|---|
| SHADE_Arena | 167 | SHADE workspace/travel/banking/spam-filter 等环境 |
| Agent_SafetyBench | 33 | Agent SafetyBench 发布环境与本地兼容环境 |
| 合计 | 200 | 混合运行 |

## 3. 运行脚本简介

### 3.1 `Tool-chaining.py`

主运行脚本：

```text
Code/FormalExp/AgentLAB-main/Tool-chaining.py
```

主要职责：

1. 读取 `filtered_top_200_attacks.json`；
2. 根据 `generation_config.dataset` 选择 SHADE_Arena 或
   Agent_SafetyBench 后端；
3. 使用 DeepSeek Flash 作为 planner、attacker 和 judge；
4. 使用本地 vLLM 中的 Qwen3-8B 作为 victim；
5. 执行多轮攻击规划和目标 Agent 工具调用；
6. 持续保存 `results.json`；
7. 结束时保存 `final_results.json`；
8. 为每个样本保存不可覆盖的 `raw_trace.json` 和 SHA256。

DeepSeek 兼容逻辑包括：

- DeepSeek endpoint 下自动使用合法的 `deepseek-*` 模型名；
- planner/judge 使用 JSON response format；
- auxiliary thinking 强制关闭；
- DeepSeek 返回 HTTP 200 但 content 为空时，使用 thinking-off 自动重试；
- API 调用包含重试机制。

### 3.2 `trace_artifacts.py`

统一 trace 写入模块：

```text
Code/FormalExp/AgentLAB-main/trace_artifacts.py
```

每个样本写入：

```text
traces/<sample_id>/raw_trace.json
traces/<sample_id>/raw_trace.sha256
```

`raw_trace.json` 保留：

- task；
- attack/planner history；
- victim 模型与采样参数；
- 环境和工具定义；
- 完整 messages；
- model calls；
- tool calls 与 tool results；
- judge history 和最终分数；
- 数据来源、run ID 和 Agent_SafetyBench 映射 provenance；
- `label_status=pending`。

Trace 采用 create-once 语义：同一个 sample identity 如果内容不同，脚本拒绝覆盖，
防止静默损坏已有实验数据。

## 4. 正式模型配置

### 4.1 Victim

| 项目 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-8B` |
| 权重目录 | `/root/autodl-tmp/models/Qwen3-8B` |
| API | `http://127.0.0.1:8001/v1` |
| temperature | `0.7` |
| top-p | `0.8` |
| top-k | `20` |
| max tokens | `768` |
| thinking | `off` |

### 4.2 Auxiliary

| 项目 | 值 |
|---|---|
| planner/attacker/judge | `deepseek-flash` |
| endpoint | `https://api.deepseek.com` |
| thinking | `off` |

### 4.3 攻击参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `max_turns` | 10 | planner 最多优化10轮 |
| `max_agent_rounds` | 10 | 每个 planner turn 内 victim 最多执行10轮工具调用 |
| `success_threshold` | 3 | judge 最终评分达到3即记为 runner success |
| `seed` | 20261118 | 数据抽样 seed；全量运行时不负责逐请求生成随机性 |

## 5. vLLM 启动命令

在独立终端启动。正式实验实际需要将 GPU memory utilization 设置为 `0.95`；
`0.90` 在当前服务器上不足以提供所需 KV cache。

```bash
set -o pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export TC_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1/tool_chaining_deepseek_flash"
export QWEN_MODEL=/root/autodl-tmp/models/Qwen3-8B

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$TC_ROOT"

CUDA_VISIBLE_DEVICES=0 vllm serve "$QWEN_MODEL" \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.95 \
  --max-model-len 40960 \
  --max-num-seqs 20 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  2>&1 | tee "$TC_ROOT/vllm.log"
```

说明：

- `--gpu-memory-utilization 0.95` 是本实验实际使用值；
- `--max-model-len 40960` 是输入与输出合计的上下文上限；
- `--max-num-seqs 20` 是 vLLM 服务容量，不等于 Tool Chaining 同时并行20个
  benchmark 样本；当前 `Tool-chaining.py` 的样本主循环仍是顺序执行；
- `--reasoning-parser qwen3` 只负责解析 reasoning，不负责开启 thinking；
- benchmark 请求还会显式传入 `--victim_thinking off`。

另一个终端检查服务：

```bash
curl -f http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/v1/models
```

模型列表应包含：

```text
Qwen/Qwen3-8B
```

## 6. 公共实验环境

以下 smoke 和 full 命令均先设置：

```bash
set -o pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export AL_ROOT="$SD_ROOT/Code/FormalExp/AgentLAB-main"
export TC_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1/tool_chaining_deepseek_flash"

export VICTIM_MODEL_NAME=Qwen/Qwen3-8B
export VICTIM_BASE_URL=http://127.0.0.1:8001/v1

export DEEPSEEK_API_KEY='替换成你的DeepSeek密钥'
export AUX_BASE_URL=https://api.deepseek.com
export AUX_MODEL=deepseek-flash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$AL_ROOT"
curl -f http://127.0.0.1:8001/health
```

本实验不需要：

- OpenAI API key；
- Google API key；
- WebShop；
- Mem0。

若日志出现 `google-generativeai not installed`，在只使用 Qwen victim 和
DeepSeek auxiliary 时可以忽略。

## 7. Agent_SafetyBench 映射预检

正式运行前建议先做不调用任何模型 API 的33条映射检查：

```bash
export PREFLIGHT_ROOT="$TC_ROOT/preflight"
mkdir -p "$PREFLIGHT_ROOT"

python -u Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --dataset data/filtered_top_200_attacks.json \
  --dataset_source Agent_SafetyBench \
  --resolve_agent_safety_only \
  2>&1 | tee "$PREFLIGHT_ROOT/agent_safety_resolution.log"
```

正常结尾：

```text
Resolved 33 Agent-SafetyBench attacks.
```

## 8. Smoke 实验

建议 smoke 分别覆盖两个后端，各跑2条。不要只抽混合数据集2条，否则可能随机抽到
同一个后端，无法验证另一套环境。

### 8.1 SHADE_Arena smoke

```bash
export RUN_ROOT="$TC_ROOT/smoke/shade_arena"
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/traces"

python -u Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url "$VICTIM_BASE_URL" \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_api_key "$DEEPSEEK_API_KEY" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --dataset_source SHADE_Arena \
  --num_samples 2 \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --output_dir "$RUN_ROOT/results" \
  --trace_dir "$RUN_ROOT/traces" \
  2>&1 | tee "$RUN_ROOT/run.log"
```

### 8.2 Agent_SafetyBench smoke

```bash
export RUN_ROOT="$TC_ROOT/smoke/agent_safety_bench"
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/traces"

python -u Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url "$VICTIM_BASE_URL" \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_api_key "$DEEPSEEK_API_KEY" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --dataset_source Agent_SafetyBench \
  --num_samples 2 \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --output_dir "$RUN_ROOT/results" \
  --trace_dir "$RUN_ROOT/traces" \
  2>&1 | tee "$RUN_ROOT/run.log"
```

### 8.3 Smoke 验收

```bash
for backend in shade_arena agent_safety_bench; do
  RUN_ROOT="$TC_ROOT/smoke/$backend"
  echo "===== $backend ====="
  find "$RUN_ROOT/traces" -name raw_trace.json | wc -l
  find "$RUN_ROOT/traces" -name raw_trace.sha256 | wc -l
  test -s "$RUN_ROOT/results/final_results.json" && echo "final_results.json OK"
done
```

两个后端均应显示2份 `raw_trace.json`、2份 `raw_trace.sha256`，并存在非空的
`final_results.json`。

## 9. 一次性200条正式实验

这是当前推荐命令。关键点：

- 不传 `--dataset_source`，因此同时运行两个后端；
- 不传 `--num_samples`，因此运行全部200条；
- 结果和 trace 从开始就位于同一个正式目录，无需后续整合。

```bash
set -o pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export AL_ROOT="$SD_ROOT/Code/FormalExp/AgentLAB-main"
export TC_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1/tool_chaining_deepseek_flash"
export RUN_ROOT="$TC_ROOT/full"

export VICTIM_MODEL_NAME=Qwen/Qwen3-8B
export VICTIM_BASE_URL=http://127.0.0.1:8001/v1

export DEEPSEEK_API_KEY='替换成你的DeepSeek密钥'
export AUX_BASE_URL=https://api.deepseek.com
export AUX_MODEL=deepseek-flash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/traces"
cd "$AL_ROOT"

curl -f http://127.0.0.1:8001/health

python -u Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url "$VICTIM_BASE_URL" \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_api_key "$DEEPSEEK_API_KEY" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --output_dir "$RUN_ROOT/results" \
  --trace_dir "$RUN_ROOT/traces" \
  2>&1 | tee "$RUN_ROOT/run.log"
```

注意：不要在已经完成或失败过的 `full/` 目录中直接复跑。不可变 trace 不允许
同 identity 被不同内容覆盖。需要复跑时应使用 `full_r2/`、`full_r3/` 等新目录。

## 10. 正式实验完整性检查

```bash
python3 - "$RUN_ROOT" "$AL_ROOT/data/filtered_top_200_attacks.json" <<'PY'
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
result_dir = root / "results"

dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
results = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
final = json.loads((result_dir / "final_results.json").read_text(encoding="utf-8"))
traces = []
checksum_errors = []

for path in sorted((root / "traces").glob("*/raw_trace.json")):
    trace = json.loads(path.read_text(encoding="utf-8"))
    traces.append(trace)
    checksum_path = path.with_name("raw_trace.sha256")
    if not checksum_path.exists():
        checksum_errors.append(f"{path.parent.name}: missing checksum")
        continue
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        checksum_errors.append(f"{path.parent.name}: checksum mismatch")

expected_ids = {str(item["original_id"]) for item in dataset}
trace_ids = [str(item["task"]["original_id"]) for item in traces]
result_ids = [str(item["attack_id"]) for item in results["results"]]
final_ids = [str(item["attack_id"]) for item in final["results"]]

duplicate_trace_ids = [
    item_id for item_id, count in Counter(trace_ids).items() if count > 1
]
missing = sorted(expected_ids - set(trace_ids), key=int)
extra = sorted(set(trace_ids) - expected_ids, key=int)

print("dataset rows:      ", len(dataset))
print("raw traces:        ", len(traces))
print("unique trace IDs:  ", len(set(trace_ids)))
print("results rows:      ", len(result_ids))
print("final result rows: ", len(final_ids))
print("missing IDs:       ", missing)
print("extra IDs:         ", extra)
print("duplicate IDs:     ", duplicate_trace_ids)
print("checksum errors:   ", checksum_errors)
print("metrics:           ", final.get("metrics"))

ok = (
    len(dataset) == 200
    and len(traces) == 200
    and len(set(trace_ids)) == 200
    and len(result_ids) == 200
    and len(final_ids) == 200
    and expected_ids == set(trace_ids) == set(result_ids) == set(final_ids)
    and not missing
    and not extra
    and not duplicate_trace_ids
    and not checksum_errors
)

if not ok:
    raise SystemExit("FAIL: Tool Chaining结果不完整")

print("PASS: 200条Tool Chaining结果完整且一致")
PY
```

## 11. 输出目录说明

正式目录结构：

```text
full/
├── run.log
├── results/
│   ├── config.json
│   ├── results.json
│   └── final_results.json
└── traces/
    ├── agentlab-.../
    │   ├── raw_trace.json
    │   └── raw_trace.sha256
    └── ...
```

文件含义：

- `run.log`：完整控制台输出；
- `config.json`：已脱敏的模型、采样、攻击和 trace 配置；
- `results.json`：运行过程中持续更新的结果；
- `final_results.json`：最终结果、指标和配置；
- `raw_trace.json`：SD_Guard 后续规则、标签和轨迹分析的权威输入；
- `raw_trace.sha256`：对应 trace 的完整性校验。

## 12. 历史正式结果

本地已整理的历史正式结果位于：

```text
results/formal/tool_chaining_deepseek_flash/full
```

它物理上来自两个源执行：

- SHADE_Arena 167条，run ID `20261118_195211_209616`；
- Agent_SafetyBench 33条，run ID `20261118_215846_625532`。

整理后：

- `traces=200`；
- `results.json=200`；
- `final_results.json=200`；
- 200个唯一 original ID；
- 0缺失、0重复、0 checksum error；
- 原始167条结果与 trace 重建值逐字段一致。

历史观测指标：

| 指标 | 值 |
|---|---:|
| 成功 | 191/200 |
| ASR | 95.5% |
| 平均分 | 2.89 |
| 分数0 | 4 |
| 分数1 | 5 |
| 分数3 | 191 |

这些是历史观测值，不是复跑必须完全相同的验收条件。Victim temperature 为0.7，
DeepSeek planner 也可能产生随机性；`--seed 20261118` 主要控制数据抽样，不是所有
模型请求的 generation seed。

历史数据另有两项注意：

- `original_id=34` 发生 victim 上下文溢出，应重跑或作为技术异常处理；
- 8条 Agent_SafetyBench 样本存在功能等价的源环境并列候选，所选来源和并列候选
  均已保存在 trace provenance 中。

## 13. 常见问题

### DeepSeek 报“passed gpt-5.1”

说明运行的是旧代码或没有传 `--attacker_model deepseek-flash`。当前命令已显式传入，
更新后的代码也会在 DeepSeek endpoint 下自动纠正不兼容模型名。

### 连续出现 `Planner returned an empty prompt`

偶发一次可记录观察；连续出现说明该样本 planner 没有生成有效攻击提示，不能把该
样本当作正常完整结果。当前代码会在 DeepSeek 200 但 content 为空时额外使用
thinking-off 重试一次。若仍连续发生，应停止该轮并使用新目录复跑。

### 为什么只看到 DeepSeek HTTP 200，没有 Environment

DeepSeek 请求用于 planner/judge。只有 planner 生成有效 prompt 后 victim 才会调用
本地 vLLM，并随后出现环境工具返回。单独的 HTTP 200 不等于已经生成完整 victim
trajectory。

### `--max-num-seqs 20` 是否表示同时跑20个样本

不是。它是 vLLM 服务最多可调度的序列数；当前 Tool Chaining 主脚本顺序处理样本。
它主要为服务容量和后续并行 runner 预留空间。

### 标签是否必须在运行 trace 时生成

不必须。当前 trace 的 `label_status` 为 `pending`，完整轨迹先保存；后续可由
SD_Guard 单独生成 episode label、row label 或规则标签。
