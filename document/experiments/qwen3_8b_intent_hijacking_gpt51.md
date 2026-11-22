# Qwen3-8B Intent Hijacking：当前固定配置与运行指南

本文对应当前仓库中的固定运行入口：

```text
/root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main/run_qwen3_8b.py
```

运行时只需调用该入口；服务器路径、数据集、输出根目录、离线 Hugging Face
设置和模型参数均已固化在 `qwen3_8b_official_profile.py`，不用在每次实验前
重复 `export SD_ROOT=...` 一类变量。

## 1. 实验构成

Intent Hijacking 是多轮纯对话劫持实验：本地 attacker 生成社工对话策略，
Qwen3-8B 作为受害 agent；外部辅助模型负责 strategy planner 与最终 judge。

| 角色 | 当前配置 | 服务地址 |
|---|---|---|
| Victim | `Qwen/Qwen3-8B`，本地权重 `/root/autodl-tmp/models/Qwen3-8B` | `http://127.0.0.1:8001/v1` |
| Attacker | `huihui-ai/Huihui-Qwen3-14B-abliterated-v2`，本地权重 `/root/autodl-tmp/models/Huihui-Qwen3-14B-abliterated-v2` | `http://127.0.0.1:8000/v1` |
| Planner / judge | `gpt-5.1`，通过 ChatAnywhere OpenAI-compatible API | `https://api.chatanywhere.tech/v1` |
| 数据集 | `data/filtered_top_200_attacks.json` | 固定在 AgentLAB 目录 |

固定的 victim sampling 参数为：temperature `0.7`、top-p `0.8`、top-k
`20`、max tokens `768`、thinking `off`。Intent runner 使用 `max_turns=7`、
`num_strategies=2`、success threshold `4/5` 和顺序执行。

## 2. 先确认硬件是否满足要求

这一步很重要。Intent 和 Memory 不一样：它必须让 **victim vLLM** 与
**attacker vLLM** 同时在线，分别占用端口 8001 和 8000。

当前 AutoDL 机器实测为一张 24GB RTX 4090；现有 Qwen3-8B vLLM 已占用约
23GB。未量化的 Huihui Qwen3-14B 权重目录约 28GB，单卡 24GB 连 attacker
自身都放不下，更不可能与 victim 同卡运行。因此，**不能在当前这台单卡
4090 上直接启动本实验**。

可行条件是：为 victim 和 attacker 提供互不争用的显存，例如两张显存充足
的 GPU（attacker 的设备通常需要约 40GB 以上并预留 KV cache；48GB 更稳妥），
或改用经过验证的量化 attacker 配置。后一种会改变当前固定 profile，不能和
本指南中的结果直接混在一起。

在满足硬件条件的服务器上，先检查：

```bash
nvidia-smi
```

下文的例子假定 victim 使用 GPU 0，attacker 使用 GPU 1。请按实际 GPU
编号替换；不要把两个服务指向同一张显存不足的卡。

## 3. 一次性配置 API 密钥

密钥只保存在被 Git 忽略的本地文件中，不写入 shell 历史、文档或 profile。
当前仓库已将它加入 `.gitignore`。

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
cp -n config/local_secrets.env.example config/local_secrets.env
nano config/local_secrets.env
chmod 600 config/local_secrets.env
```

至少填写：

```dotenv
OPENAI_API_KEY=你的ChatAnywhere真实密钥
OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
```

`OPENAI_BASE_URL` 实际上已有 profile 默认值，写在该文件中只是让所用 provider
更明确。`GPT51_MODEL = "gpt-5.1"` 位于
`qwen3_8b_official_profile.py`，不是密钥文件中的配置项。

可用下列命令确认密钥文件仍未被 Git 跟踪：

```bash
cd /root/autodl-tmp/Code/SD_Guard
git check-ignore -v Code/FormalExp/AgentLAB-main/config/local_secrets.env
```

## 4. 启动两个本地 vLLM 服务

以下每段放在一个独立终端，并保持运行。使用绝对路径，因此无需 `conda activate`
或预先 `export` 路径变量。

### 终端 A：victim（端口 8001）

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/conda-envs/sd4-vllm-cu130/bin/vllm serve \
  /root/autodl-tmp/models/Qwen3-8B \
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
  --generation-config vllm \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

### 终端 B：attacker（端口 8000）

以下命令只适用于 attacker GPU 显存充足的机器；当前单卡 24GB 服务器不可用。

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/conda-envs/sd4-vllm-cu130/bin/vllm serve \
  /root/autodl-tmp/models/Huihui-Qwen3-14B-abliterated-v2 \
  --served-model-name huihui-ai/Huihui-Qwen3-14B-abliterated-v2 \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 38000 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

服务启动完成后，在第三个终端检查两个端口和模型 ID：

```bash
curl -f http://127.0.0.1:8001/health
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8000/v1/models
```

输出模型列表必须分别含有：

```text
Qwen/Qwen3-8B
huihui-ai/Huihui-Qwen3-14B-abliterated-v2
```

## 5. 先预览，不创建输出

launcher 会解析固定 profile。`--dry-run` 不创建运行目录、不写 trace，也不检查
服务健康状态，适合先确认命令、模型和输出目录：

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
/root/autodl-tmp/conda-envs/sd4-vllm-cu130/bin/python \
  run_qwen3_8b.py intent smoke --run-tag intent_smoke_20261121 --dry-run
```

## 6. 运行 smoke

smoke 固定抽取 2 个样本，自动带 `--no_textgrad`，适合验证双 vLLM、
ChatAnywhere 的 GPT-5.1 planner/judge、结果写入和 raw trace。

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
/root/autodl-tmp/conda-envs/sd4-vllm-cu130/bin/python \
  run_qwen3_8b.py intent smoke --run-tag intent_smoke_20261121
```

实际输出根目录：

```text
/root/autodl-tmp/Code/SD_Guard/results/agentlab/qwen3_8b_nothink_v1/
  intent_hijacking_gpt_5_1_judge/smoke_intent_smoke_20261121/
```

每次重复运行必须换一个 `--run-tag`。launcher 会拒绝写入非空目录，避免把两次
实验的 trace 混在一起。

## 7. 运行 full

full 固定使用 200 个样本，且如果环境安装了 TextGrad，会启用 TextGrad。建议仅在
smoke 的本地服务、API preflight 和 trace 都正常后运行。

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
/root/autodl-tmp/conda-envs/sd4-vllm-cu130/bin/python \
  run_qwen3_8b.py intent full --run-tag intent_full_20261121
```

输出根目录：

```text
/root/autodl-tmp/Code/SD_Guard/results/agentlab/qwen3_8b_nothink_v1/
  intent_hijacking_gpt_5_1_judge/full_intent_full_20261121/
```

## 8. 运行中查看与结果文件

以 smoke tag 为例：

```bash
RUN=/root/autodl-tmp/Code/SD_Guard/results/agentlab/qwen3_8b_nothink_v1/intent_hijacking_gpt_5_1_judge/smoke_intent_smoke_20261121

tail -f "$RUN/run.log"
find "$RUN/traces" -name raw_trace.json | wc -l
find "$RUN/results" -maxdepth 2 -type f | sort
```

主要产物：

| 文件 | 用途 |
|---|---|
| `launcher_config.json` | launcher 固化的命令、环境及辅助 endpoint（不会存密钥） |
| `run.log` | launcher 和 runner 的完整 stdout/stderr |
| `results/Qwen_Qwen3-8B/results.json` | 运行中不断覆盖更新的中间结果 |
| `results/Qwen_Qwen3-8B/final_results.json` | 结束后的结果、统计和已去密钥的配置 |
| `results/Qwen_Qwen3-8B/report.md` | 人类可读的最终报告 |
| `results/Qwen_Qwen3-8B/results.csv` | 样本级汇总 |
| `traces/**/raw_trace.json` | 不可变的原始轨迹；复核成功标签时以它为准 |

## 9. 常见阻塞点

1. `Required local service is not healthy`：对应的 vLLM 没启动、端口错了，或服务
   尚在加载模型。先使用第 4 节的两条 `curl -f`。
2. `GPT-5.1 preflight returned ...`：检查 `config/local_secrets.env` 中的
   ChatAnywhere key、endpoint、网络和该 key 对 `gpt-5.1` 的访问权限。launcher
   会在创建结果目录前停止。
3. `Refusing to mix results in non-empty directory`：只换新的 `--run-tag`，不要
   删除、复用或覆盖旧实验目录。
4. CUDA OOM 或 attacker 起不来：这是当前单卡 24GB 机器的预期限制，不是 API 或
   runner 配置问题。需要更换为满足第 2 节条件的计算资源，或另行建立并记录新的
   量化 attacker 实验条件。

## 10. 配置位置速查

| 要改的内容 | 修改位置 | 是否影响已在运行的任务 |
|---|---|---|
| ChatAnywhere key / endpoint | `config/local_secrets.env` | 否；仅后续新进程 |
| GPT-5.1 planner/judge model ID | `qwen3_8b_official_profile.py` 的 `GPT51_MODEL` | 否；仅后续新进程 |
| victim、attacker 本地路径/端口 | `qwen3_8b_official_profile.py` | 否；并须与 vLLM 命令一致 |
| smoke/full 样本数 | `_intent_specs()` 的 `count = 2 if scale == "smoke" else 200` | 否；仅后续新进程 |
| 单样本参数 | `_intent_specs()` 中的 `max_turns`、`num_strategies`、`success_threshold` | 否；会改变实验条件，应使用新的 run tag |

