# Qwen3-8B AgentLAB 官方式 GPT-5.1 运行说明

本文是四个非 AgentDojo 子任务的新正式入口。固定服务器路径、模型路径、结果根目录、
Qwen non-thinking 采样参数、Hugging Face 离线设置和各角色模型均已写入：

```text
Code/FormalExp/AgentLAB-main/qwen3_8b_official_profile.py
```

GPT-5.1/GPT-5.2 请求通过现有 ChatAnywhere OpenAI-compatible endpoint
`https://api.chatanywhere.tech/v1` 转发。“官方式”指 AgentLAB 的模型角色与模型 ID，
不表示密钥或请求入口来自 OpenAI 官方。

不再为每次实验重复设置 `SD_ROOT`、`AL_ROOT`、`RUN_ROOT`、`QWEN_MODEL`、
`HF_HUB_OFFLINE` 或 `TRANSFORMERS_OFFLINE`。

## 1. 模型角色

| 子任务 | 攻击/规划角色 | Judge 及辅助评估角色 |
|---|---|---|
| Tool Chaining | GPT-5.1 planner | GPT-5.1 judge |
| Intent Hijacking | 本地 Huihui-Qwen3-14B attacker | GPT-5.1 planner/judge |
| Objective Drifting | GPT-5.2 attacker（保留源码默认） | GPT-5.1 judge/extractor/stealthiness |
| Memory Poisoning | GPT-5.1 planner/attacker/Mem0 LLM | GPT-5.1 judge/evasiveness judge |

四条任务的 victim 均为本地 `Qwen/Qwen3-8B`，使用 non-thinking 条件：
`temperature=0.7`、`top_p=0.8`、`top_k=20`、`max_tokens=768`。

## 2. 一次性密钥配置

仅执行一次：

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
cp -n config/local_secrets.env.example config/local_secrets.env
nano config/local_secrets.env
chmod 600 config/local_secrets.env
```

文件中只需填真实的 ChatAnywhere key：

```text
OPENAI_API_KEY=...
```

密钥不会出现在 launcher 命令、trace 或 manifest 中。

## 3. 启动 Qwen victim

Tool Chaining、Objective Drifting 和 Memory Poisoning 共用以下服务。在独立终端运行：

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
python start_qwen3_8b_vllm.py victim
```

脚本内部已经固定模型路径、端口 8001、40,960 context、GPU utilization 0.95、
tool parser、Qwen reasoning parser 和 non-thinking chat template，并自动保存服务日志。

## 4. 最短运行命令

在 runner 终端进入 AgentLAB 目录一次：

```bash
cd /root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main
```

先 smoke，验收后再 full：

```bash
python run_qwen3_8b.py tool smoke
python run_qwen3_8b.py tool full

python run_qwen3_8b.py objective smoke
python run_qwen3_8b.py objective full

python run_qwen3_8b.py memory smoke
python run_qwen3_8b.py memory full
```

launcher 会自动：

- 选择正确的 Conda Python；Memory Poisoning 自动使用独立 memory 环境；
- 检查本地 vLLM health；
- 检查固定模型、数据集和 MiniLM 路径；
- 只从 `local_secrets.env` 读取 `OPENAI_API_KEY`；
- 创建独立 GPT-5.1 结果目录并保存 `launcher_config.json`；
- 将 stdout/stderr 同时写入 `run.log`；
- 拒绝向非空目录混入新结果。

若确实需要复跑，使用新 tag，不覆盖原结果：

```bash
python run_qwen3_8b.py objective full --run-tag r2
```

只查看解析后的完整命令，不调用模型、不写文件：

```bash
python run_qwen3_8b.py objective full --dry-run
```

## 5. Intent Hijacking 的两个本地服务

Intent Hijacking 需要同时运行 Qwen victim 和官方 Huihui attacker。在两个独立终端启动：

先停止供其他任务使用的普通 `victim` 服务；`victim-for-intent` 会重新占用同一个
8001 端口。

```bash
python start_qwen3_8b_vllm.py victim-for-intent
```

```bash
python start_qwen3_8b_vllm.py intent-attacker
```

然后在第三个终端运行：

```bash
python run_qwen3_8b.py intent smoke
python run_qwen3_8b.py intent full
```

远端实机是 24GB RTX 4090，而 BF16 的 8B/14B 模型目录约为 16GB/28GB，不能直接
同时完整放入显存。两个 Intent 服务因此将 context 限为 8192，并分别使用 8GB/20GB
CPU offload；服务器可用内存已核实足够。这样保留官方 14B attacker 权重，但生成速度
会明显慢于全显存推理。正式开始前仍应确认两项 health check 都能通过；launcher 也会
再次检查。

## 6. 新结果目录

新运行不会触碰已有 DeepSeek 结果：

```text
results/agentlab/qwen3_8b_nothink_v1/
  tool_chaining_gpt_5_1/
  intent_hijacking_gpt_5_1_judge/
  objective_drifting_official_gpt_5_1_judge/
  memory_poisoning_official_gpt_5_1/
```

已有 `*_deepseek_flash` 目录继续作为历史运行保留，不应改名、覆盖或把新 trace 混进去。

## 7. 重要语义

这里恢复的是运行过程中的官方式模型角色，不是对旧 DeepSeek trace 做离线改名。
新 full 运行会产生全新的攻击过程，因为在线 judge 会影响 Tool Chaining 的停止条件，
Objective Drifting 和 Memory Poisoning 的后续优化也会受到辅助模型输出影响。

raw trace 仍保持 `label_status=pending`。GPT-5.1 的 benchmark judge 结果与后续
SD_Guard `episode_label` / `row_label` 是两层不同的数据，不应混写。
