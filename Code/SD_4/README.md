# SD_4：双 L20 vLLM + passive Guard 的完整 2×4 实验

SD_4 保留 SD_3 的数据集、逐样本 seed、thinking 条件、Guard 策略、盲评格式和统计逻辑，只把 speculative decoding 换成 vLLM 0.31 的官方 `draft_model` 路径，并把“Draft proposal 先过 Guard、再交给 Target 验证”的在线实验并入同一次 2×4 运行。

| 条件 | `target_only` | `speculative` | `draft_only` | `inline_guard` |
|---|---|---|---|---|
| `think_on` | 14B Target + posthoc Guard | vLLM Draft→Target + posthoc Guard | 0.6B Draft + posthoc Guard | vLLM Draft→Guard→Target |
| `think_off` | 14B Target + posthoc Guard | vLLM Draft→Target + posthoc Guard | 0.6B Draft + posthoc Guard | vLLM Draft→Guard→Target |

默认参数：100 条、base seed 42、每条样本 seed 为 `sha256(base_seed:sample_id) % (2**31-1)`、`max_new_tokens=4096`、K=3、Target `temperature=0.6/top_p=0.95/top_k=20`，Draft 使用 vLLM 的 greedy proposal，standard rejection sampling。两个 thinking 条件和四种模式对同一 `sample_id` 使用完全相同的 seed。

## 模型与双 GPU 约束

默认模型路径为：

```text
Draft:  /home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2
Target: /home/model_and_data/yxz/model/Huihui-Qwen3-14B-abliterated-v2
Guard:  /home/model_and_data/yxz/model/Qwen3Guard-Stream-4B
```

实际目录名不同时通过 CLI 或环境变量覆盖。父进程为每个隔离 worker
设置 `CUDA_VISIBLE_DEVICES=<ENGINE_GPU_ID>,<GUARD_GPU_ID>`，顺序是实验契约的一部分：

- 逻辑 `cuda:0`：vLLM Target 14B + Draft 0.6B，`tensor_parallel_size=1`；
- 逻辑 `cuda:1`：Transformers Qwen3Guard-Stream-4B；
- 两张卡必须不同，worker 会强制检查恰好可见两张 GPU；
- 默认 `gpu_memory_utilization=0.80` 只约束逻辑 `cuda:0` 的 vLLM engine。

当前两张 L20 的 `nvidia-smi topo -m` 路径为 `SYS`，不是 NVLink。在线 Guard
的 token handoff 实际采用 GPU0→host Python IDs→GPU1 的 staged 路径；结果同时保存
D2H、H2D、Guard 计算、RPC/控制开销。另有直接 GPU0→GPU1 `Tensor.to` 微基准，
但它只是拓扑诊断，不会冒充实际 pipeline 传输，也不进入正式延迟。

## 在线插入点与不变性

`sdguard_sd.vllm_extension.SD4WorkerExtension` 通过 vLLM 的 `worker_extension_cls` 注入实际 GPU worker。对默认 V2 runner，它在运行时包装 vLLM 0.31 的 `PlainDraftModelSpeculator.propose()`；若该 0.31 构建显式使用 V1 runner，则回退包装 `GPUModelRunner.propose_draft_token_ids()`：

```text
Draft proposal 已返回 → Guard 增量观察/报警 → 原 Draft IDs 原样返回 → Target verification
```

适配层不修改 token tensor、不修改采样状态、不参与接受/拒绝，也不提前停止。每条 `inline_guard` 紧跟同 seed 的标准 `speculative` 请求；程序逐 token 比较最终 IDs，并比较 vLLM `detailed` per-request acceptance trace。任一不一致立即失败，因此不会错误复用人工标签。

vLLM 0.31 会在 GPU rejection sampling 后先生成下一轮 proposal，再由 scheduler
应用 EOS/长度停止。异步流水可能在 scheduler 确认终止前多完成一轮或多轮内部
verification；SD_4 只在结果记录中把 verified/Guard 语义裁到
`CompletionOutput.token_ids`；原始
accepted/drafted 数仍单独保存并与 vLLM detailed metrics 核对。这个结算不回写
vLLM，也不改变输出。若两条流不是这种合法的末尾裁剪，而是在中间发生 token 分叉，
实验仍立即失败并报告首个分叉窗口。

按本次要求，SD_3 后补的离线 trace replay 不进入 SD_4；`inline_guard` 只做重新调用模型的在线生成与检测。

适配层只支持固定的 vLLM **0.31.0 base version**（允许类似 `0.31.0+cu129` 的本地构建后缀），且不修改 site-packages。升级 vLLM 时必须重新核对 hook 签名。

## Qwen3Guard / Transformers 5.17 兼容层

Guard checkpoint 的配置记录为 Transformers 4.55，并通过 `trust_remote_code`
加载自定义 Qwen3Guard 实现；当前 vLLM 0.31 环境固定使用 Transformers 5.17。
SD_4 在 Guard worker 内集中处理已知 API 差异，不改模型目录、Hugging Face cache
或 `site-packages`：

- 从 tokenizer 补齐 5.17 稀疏配置中缺失的 `pad_token_id`，并校验 `layer_types`；
- 恢复旧 Qwen3Guard 查询的标准 `default` RoPE 初始化入口；
- 将旧 Mask API 的 `input_embeds`/`cache_position` 调用翻译为 5.17 的
  `inputs_embeds`/cache-derived offset；
- 使用 `dtype` 加载参数，避免依赖已废弃的 `torch_dtype` 调用；
- 使用负数 `tokens_to_remove` 回滚 DynamicCache，不使用已废弃的正数绝对长度；
- posthoc Guard 不再手工构造跨 backend 含义不一致的 bool 4-D mask，由模型生成
  与当前 attention backend 匹配的 causal mask。

每次 Guard 加载后会在模型加载计时阶段自动执行一次真实 GPU 兼容自检：两 token
prefill、单 token 增量 forward、四个分类 head 的 shape、DynamicCache 长度增长及
回滚全部通过后，worker 才会开始数据集实验。自检结果写入 worker runtime metadata
的 `guard_adapter.compatibility`；自检时间属于模型加载时间，不进入正式样本延迟。

## 延迟口径

正式结果是 100 条、batch size 1、关闭 prefix cache、warm-up、模型加载和启动时
跨卡微基准均排除的 wall latency：

- `target_only`：Target 模型生成、posthoc Guard、Guard input H2D、raw pipeline、transfer-excluded pipeline；
- `draft_only`：Draft 模型生成、posthoc Guard、Guard input H2D、raw pipeline、transfer-excluded pipeline；
- `speculative`：vLLM SD、posthoc Guard、Guard input H2D、raw pipeline、transfer-excluded pipeline；
- `inline_guard`：Guard prompt prefill、proposal、commit、proposal/commit D2H+H2D staged handoff、raw operational pipeline、transfer-excluded pipeline，以及与同样本标准 SD 的 raw/adjusted paired overhead；最终 posthoc Guard 只是审计，不计入 operational latency。

`raw` wall latency 始终是正式主结果；`transfer-excluded` 只是回答“去掉不同放置造成的
传输后计算路径如何比较”的辅助结果，不覆盖或修改原始测量。

每项正式时延汇总 mean、median、P90、P95、min、max。SD 内部阶段使用完全独立的
轻量诊断运行：适配层在固定 vLLM 0.31 的 Target forward、sampling/rejection 和
Draft proposal 调用边界做 CUDA 同步 wall 计时，Guard 继续记录 prefill、proposal、
commit 及传输时间。诊断只写小型 `stage_timing_summary.json`，不会启用 PyTorch
Profiler，也不会生成、压缩或解析 `.pt.trace.json(.gz)`。同步会扰动正常流水，
所以内部阶段结果只用于拆分诊断，不混入正式 100 条端到端统计。

## 运行

当前两张 L20 使用环境：

```text
/home/model_and_data/yxz/envs/sd-vllm-cu129
Python 3.10.22
vLLM 0.31.0+cu129
PyTorch 2.13.0+cu129
Transformers 5.17.0
Tokenizers 0.23.3
```

不要使用 `/home/model_and_data/yxz/envs/SD_Guard`：它没有 vLLM，且 Torch
2.9.1+cu128/Transformers 4.57.6 不属于当前固定适配栈。先补齐实验依赖：

```bash
source /home/model_and_data/yxz/envs/sd-vllm-cu129/bin/activate
python -m pip install \
  'datasets==4.8.5' 'accelerate==1.15.0' \
  'safetensors==0.8.0' 'tqdm==4.70.1'
python -m pip check
python scripts/check_l20_cu129_env.py \
  --output sd4-l20-cu129-env.json
```

在启动 14B Target 前，可先只暴露物理 GPU1 并单独验证 Guard 兼容层；该命令会
实际加载 4B Guard 并执行上述 prefill/incremental/cache rollback 自检：

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/SD_4
CUDA_VISIBLE_DEVICES=1 python scripts/check_guard_compat.py \
  --device cuda:0 \
  --local-files-only
```

成功时 JSON 中必须包含 `"status": "passed"`。这里的逻辑 `cuda:0` 是只暴露
物理 GPU1 后的重新编号；完整 SD4 worker 中 Guard 仍是逻辑 `cuda:1`。

不要在这个环境中执行裸 `pip install vllm`，否则可能把 `0.31.0+cu129` 替换为
默认 CUDA 13.0 artifact。`INSTALL_CUDA130.md` 只保留给 CUDA 13/R580 服务器，
不适用于当前驱动 550 的 L20 主机。

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/SD_4

ENGINE_GPU_ID=0 \
GUARD_GPU_ID=1 \
OUTPUT_DIR=/home/model_and_data/yxz/Code/SD_Guard/SD_4/outputs/sd4_2x4_100 \
bash scripts/run_matrix.sh --local-files-only
```

也可直接运行：

```bash
python run_matrix.py \
  --output-dir /path/to/output \
  --engine-gpu-id 0 \
  --guard-gpu-id 1 \
  --local-files-only
```

正式 100 条之前先跑完整双卡链路 smoke test：

```bash
SAMPLE_COUNT=1 \
MAX_NEW_TOKENS=64 \
MAX_MODEL_LEN=4096 \
ENGINE_GPU_ID=0 \
GUARD_GPU_ID=1 \
OUTPUT_DIR="$PWD/outputs/sd4_l20_smoke" \
bash scripts/run_matrix.sh \
  --local-files-only \
  --overwrite
```

该 smoke test 必须同时通过模型放置、普通 SD/inline 输出一致、acceptance trace
一致和 Guard 双卡 handoff，之后再恢复默认 100 条、4096 output tokens。

异步调度下，GPU 可能在请求结束后多完成一轮验证。`inline_guard_round_records.jsonl`
会保留这种轮次作实际执行工作诊断（`commit_kind=terminal_output_clipped`、
`scheduler_metric_included=false`），但它不属于 vLLM 返回的逐请求 acceptance
指标，也不进入最终生成文本或 Guard 判定。延迟仍计入实际执行耗时。

一条命令依次启动 Target、Draft、speculative 三个隔离进程，避免不同 vLLM engine 的 CUDA allocator/KV cache 相互污染；这不是三张 GPU，所有进程串行复用同一物理 GPU。`speculative` 与 `inline_guard` 在同一个 engine/Guard 实例中成对运行。

正式矩阵默认且固定为无 Profiler、无同步插桩。各阶段延迟另跑一次轻量诊断；它只
加载 speculative engine，使用 manifest 中第一条样本，分别测 think_on/off 的标准
SD 和 inline SD：

```bash
SAMPLE_COUNT=1 \
WARMUP_SAMPLES=1 \
MAX_NEW_TOKENS=2048 \
STAGE_TIMING_MAX_NEW_TOKENS=32 \
ENGINE_GPU_ID=0 \
GUARD_GPU_ID=1 \
OUTPUT_DIR="$PWD/outputs/sd4_stage_timing" \
bash scripts/run_stage_timing.sh \
  --local-files-only \
  --sample-manifest "$PWD/outputs/sd4_l20_smoke_fix2/sample_manifest.json" \
  --overwrite
```

`WARMUP_SAMPLES` 在阶段测速模式中表示每个 think 模式正式计时前，完整预热
standard SD 和 inline SD 的次数，默认 1。预热会覆盖 Target、Draft、rejection
sampling、Guard 及跨设备交接，并检查两条路径的输出与 acceptance trace 一致；
预热调用不会开启阶段计时 session，也不会进入任何 `count/mean/p90/total`。

`--profile-vllm` 只保留为旧命令兼容别名，现在同样进入无 trace 的
`--stage-timing-only`；代码中已不再配置 vLLM Torch profiler。

## 输出

```text
<matrix>/sample_manifest.json
<matrix>/run_metadata.json
<matrix>/think_{on,off}/{target_only,speculative,draft_only,inline_guard}/
  run_config.json
  raw/generation_records.jsonl
  raw/guard_token_records.jsonl
  raw/sd_round_records.jsonl
  raw/inline_guard_token_records.jsonl
  raw/inline_guard_round_records.jsonl
  raw/latency_summary.json
  blind/blind_outputs.jsonl
  private/original_dataset_audit.jsonl
<matrix>/analysis/matrix_summary.json
<matrix>/analysis/matrix_guard_summary.csv
<matrix>/analysis/latency_report.json
```

轻量诊断另行输出：

```text
<diagnostic>/think_{on,off}/{speculative,inline_guard}/raw/stage_timing_summary.json
<diagnostic>/diagnostics/stage_timing/stage_timing_index.json
<diagnostic>/analysis/stage_timing_report.json
```

生成完成后按 `JUDGING_GUIDE.md` 给六个非 inline cell 做盲评，并把 `judge_labels.jsonl` 放进对应 `blind/` 目录。随后运行：

```bash
MATRIX_DIR=/path/to/output bash scripts/analyze_matrix.sh
```

脚本会再次逐 token 核验两组 `speculative`/`inline_guard` 输出及保存的 acceptance-trace 等价证明，通过后才复制两份 speculative 标签给 inline cell；然后统一计算八组 accuracy、recall、FNR、FPR、F1 并重建矩阵汇总。
