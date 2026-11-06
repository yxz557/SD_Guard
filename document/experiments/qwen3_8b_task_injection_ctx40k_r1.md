# Qwen3-8B AgentDojo Task-Injection 全量 Trace 实验记录

## 1. 记录目的

本文记录 `qwen3-8b-task-injection-ctx40k-r1` 全量实验的环境、服务参数、采样参数来源、运行命令和已知限制，供后续在相同基线上补跑同一配置的多条 trace、构造分类器训练数据和开展可重复性实验时参考。

本文严格区分：

- **显式配置**：当时的 shell 命令或 AgentDojo 请求中直接指定的值；
- **继承/推定配置**：未在请求中指定，由 vLLM、模型目录中的 `generation_config.json` 或 Qwen3 chat template 默认值决定；
- **未来建议**：后续复跑时建议显式锁定的值，不代表原命令中曾经显式传入。

## 2. 实验身份与路径

| 项目 | 值 |
|---|---|
| 模型 | `/root/autodl-tmp/models/Qwen3-8B` |
| Benchmark | AgentDojo Task Injection |
| Benchmark version | `v1.2.1` |
| Pipeline | `vllm_parsed` |
| Suite | `full` |
| Task pairs | `full` |
| Attack | `full` |
| Run ID | `qwen3-8b-task-injection-ctx40k-r1` |
| Benchmark/sample seed | `20261030` |
| 并发 worker | `20` |
| vLLM 端口 | `8000` |
| 原始日志目录 | `$AD_REPO/runs/qwen3_8b_task_injection_ctx40k_r1` |
| 结果目录 | `/root/autodl-tmp/Code/SD_Guard/results/agentdojo/task_injection_ctx40k_r1` |
| Conda 环境 | `/root/autodl-tmp/conda-envs/sd4-vllm-cu130` |

本地结果统计记录：全量共 4,745 条，4,709 条有效，其中 613 条攻击成功，整体 ASR 为 13.02%。

## 3. Hugging Face Mirror 与离线模式

HF Mirror 地址记录为：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

实际运行使用本地模型，并启用 Hugging Face/Transformers 离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

注意：

- `HF_ENDPOINT` 用于在线下载或补齐模型文件时访问 HF Mirror；
- 当 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 时，离线模式优先，运行过程不会使用 Mirror 下载文件；
- 因此实际跑 benchmark 前必须确认模型、tokenizer、chat template 和 `generation_config.json` 已完整存在于本地模型目录。

如需通过 Mirror 准备模型，应在单独的在线准备阶段关闭离线模式：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
```

模型准备完成后，再恢复离线设置开展正式实验。

## 4. vLLM 服务参数

### 4.1 当时显式指定的参数

| 参数 | 值 | 说明 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0` | 使用 GPU 0 |
| `--host` | `127.0.0.1` | 仅本机访问 |
| `--port` | `8000` | OpenAI-compatible API 端口 |
| `--tensor-parallel-size` | `1` | 单卡张量并行 |
| `--dtype` | `auto` | 由模型/vLLM 自动选择 dtype |
| `--gpu-memory-utilization` | `0.90` | KV cache 等最多使用约 90% GPU 显存 |
| `--max-model-len` | `40960` | 最大总上下文长度 40,960 tokens |
| `--max-num-seqs` | `20` | 最多并发处理 20 条序列 |
| `--trust-remote-code` | 开启 | 允许加载模型仓库自定义代码 |
| `--enable-auto-tool-choice` | 开启 | 开启自动工具选择 |
| `--tool-call-parser` | `hermes` | 解析模型输出中的工具调用 |
| `--reasoning-parser` | `qwen3` | 将 Qwen3 reasoning 与普通 content 分离 |

`--max-model-len 40960` 是“输入 + 生成”的总上下文上限，不等价于单独允许生成 40,960 个新 token。原始 AgentDojo 请求没有显式传入 `max_tokens`，实际可生成长度由剩余上下文和 vLLM 行为共同决定。

### 4.2 当时没有显式指定的生成参数

AgentDojo 的 OpenAI-compatible 请求没有显式传递 `top_p`、`top_k`、`min_p` 或生成 seed。代码虽然把 `temperature` 的默认值写为 `0.0`，但发送请求时使用：

```python
temperature=temperature or NOT_GIVEN
```

因此 `0.0` 被当作 false，最终没有向 vLLM 发送 temperature，服务端继续使用自己的生成配置。

当时也没有给 `vllm serve` 指定 `--generation-config`。vLLM 默认使用 `auto`，即从本地模型目录加载 `generation_config.json`。

如果本地 `/root/autodl-tmp/models/Qwen3-8B/generation_config.json` 与官方 Qwen3-8B 文件一致，则本次实验的推定有效生成配置为：

| 参数 | 推定有效值 | 来源/可信度 |
|---|---:|---|
| `do_sample` | `true` | Qwen3-8B `generation_config.json` |
| `temperature` | `0.6` | Qwen3-8B `generation_config.json` |
| `top_p` | `0.95` | Qwen3-8B `generation_config.json` |
| `top_k` | `20` | Qwen3-8B `generation_config.json` |
| `min_p` | `0`/默认 | Qwen3 thinking-mode 推荐设置；原请求未显式传入 |
| generation seed | 未显式指定 | benchmark 的 `--seed` 不是逐请求生成 seed |

以上值属于**高可信推定值，不是原命令中的显式参数**。复现实验前必须在远端核验并记录本地文件：

```bash
sed -n '1,160p' /root/autodl-tmp/models/Qwen3-8B/generation_config.json
sha256sum /root/autodl-tmp/models/Qwen3-8B/generation_config.json
vllm --version
```

### 4.3 Thinking 配置

当时没有显式传递 `enable_thinking`。原版 Qwen3-8B chat template 默认：

```text
enable_thinking = true
```

因此本次实验大概率运行在 thinking mode。`--reasoning-parser qwen3` 负责解析 thinking，不是开启 thinking 的开关。

需要注意，当前 AgentDojo 的 `ChatAssistantMessage` 和 OpenAI 响应转换代码只保留 `content` 与 `tool_calls`，未保存 vLLM 返回的 `reasoning`/`reasoning_content`。所以模型运行时可能生成了 thinking，但现有 `raw_trace.json` 主要是完整的**行为轨迹**，不包含完整的显式 reasoning 文本。

## 5. 原始 vLLM 启动命令

以下保留原始运行逻辑，并补充记录 HF Mirror 环境变量。由于启用了离线模式，Mirror 在正式运行阶段不会被实际访问。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export AD_MODEL=/root/autodl-tmp/models/Qwen3-8B
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 vllm serve "$AD_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 40960 \
  --max-num-seqs 20 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

## 6. 原始 AgentDojo 全量运行命令

在另一个终端执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export AD_MODEL=/root/autodl-tmp/models/Qwen3-8B
export AD_REPO=/root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main/Task-Injection/agentdojo
export AD_LOG_ROOT="$AD_REPO/runs/qwen3_8b_task_injection_ctx40k_r1"
export AD_RESULT_ROOT=/root/autodl-tmp/Code/SD_Guard/results/agentdojo/task_injection_ctx40k_r1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_LLM_PORT=8000

cd "$AD_REPO"

curl -f http://127.0.0.1:8000/health

bash scripts/run_qwen3_8b_task_injection.sh \
  --suite full \
  --pairs full \
  --attack full \
  --workers 20 \
  --seed 20261030 \
  --benchmark-version v1.2.1 \
  --target-model-id "$AD_MODEL" \
  --log-root "$AD_LOG_ROOT" \
  --result-root "$AD_RESULT_ROOT" \
  --run-id qwen3-8b-task-injection-ctx40k-r1
```

## 7. Seed 的含义

命令中的：

```bash
--seed 20261030
```

是运行脚本的 benchmark/sample seed，主要用于抽样 task pairs、manifest 命名和结果 provenance。由于本次使用 `--pairs full`，所有 task pairs 都会运行，该 seed 不负责逐条 LLM 生成采样。

因此不能把它理解为“同一配置重复运行时一定产生同一条 trace”。当前服务启动命令和 OpenAI 请求都没有显式提供逐请求 generation seed；加上 `temperature=0.6`、`top_p=0.95`、`top_k=20` 的推定采样配置，同一任务配置重复运行可能产生不同轨迹和不同攻击结果。

## 8. 后续多 Trace 实验的复现建议

### 8.1 建议显式锁定服务端行为

为了避免未来 vLLM 版本或模型目录变化导致隐式默认值漂移，后续复跑建议在保留原始模型配置的同时，显式覆盖关键采样参数与 thinking 模式：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$AD_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 40960 \
  --max-num-seqs 20 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --generation-config auto \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --default-chat-template-kwargs '{"enable_thinking":true}'
```

该命令是**未来建议的锁定版**，不是原始命令的逐字复刻。正式大规模复跑前应先用一条任务验证当前安装的 vLLM 版本支持这些参数。

更严格的做法是在 AgentDojo 的每次 OpenAI 请求中显式传递 `temperature`、`top_p`、`top_k`、`chat_template_kwargs` 和逐请求 seed，并将这些值写入每条 trace 的 provenance。

### 8.2 每个重复轮次使用独立目录与 Run ID

不要复用 r1 的日志和结果目录。示例 r2：

```bash
export AD_LOG_ROOT="$AD_REPO/runs/qwen3_8b_task_injection_ctx40k_r2"
export AD_RESULT_ROOT=/root/autodl-tmp/Code/SD_Guard/results/agentdojo/task_injection_ctx40k_r2
```

并相应设置：

```bash
--run-id qwen3-8b-task-injection-ctx40k-r2
```

后续依次使用 `r3`、`r4`、`r5`，防止已存在结果被跳过、覆盖或与不同轮次混淆。

### 8.3 多 Trace 实验必须保持不变的项目

- 模型权重与 tokenizer 文件；
- 模型目录中的 `generation_config.json`，或显式锁定后的采样参数；
- Qwen3 thinking 模式；
- AgentDojo benchmark version `v1.2.1`；
- suite、task pair 和 attack 配置；
- system prompt、tool schema、注入文本与执行环境初始状态；
- vLLM、Transformers、OpenAI SDK 和 AgentDojo 版本；
- `max-model-len=40960`；
- trace 导出 schema 与标签计算逻辑。

允许/需要变化的项目：

- 重复轮次编号；
- 独立的 log/result root 和 run ID；
- 逐请求 generation seed（若目标是获得相同配置下的独立随机 rollout）；
- 由随机采样产生的模型 reasoning、tool call 和最终结果。

### 8.4 正式复跑前的记录清单

```bash
date -Iseconds
nvidia-smi
vllm --version
python -c "import transformers, openai; print('transformers', transformers.__version__); print('openai', openai.__version__)"
sha256sum "$AD_MODEL/generation_config.json"
sha256sum "$AD_MODEL/tokenizer_config.json"
git -C "$AD_REPO" rev-parse HEAD
curl -f http://127.0.0.1:8000/health
```

同时建议保存 vLLM 启动日志，以确认它实际加载的 generation config、chat template、模型最大长度和 parser 设置。

## 9. 当前已知限制

1. 原始运行没有显式锁定 generation seed，因此不能仅凭 `--seed 20261030` 重建完全相同的 token 序列。
2. temperature、top-p、top-k 是根据 vLLM 默认 `generation-config=auto` 和官方 Qwen3-8B 配置推定的；必须核验远端本地模型文件才能最终确认。
3. thinking 大概率默认开启，但当前 trace 导出没有保存 `reasoning`/`reasoning_content`。
4. 如果后续修改 trace schema 以保存 reasoning，应提高 schema version，并区分“旧版行为 trace”和“新版含 reasoning trace”。
5. `workers=20` 和 `max-num-seqs=20` 会影响吞吐与调度；如果追求逐请求严格可复现，还应记录并控制请求顺序和 request-level seed。

