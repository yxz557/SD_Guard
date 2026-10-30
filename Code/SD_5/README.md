# SD_5：Transformers 5.17 + vLLM + passive Guard 的完整 2×4 实验

SD_5 保留 SD_4 的实验设计，并加入 Qwen3Guard custom code 对 Transformers 5.17 的兼容层：补齐 `pad_token_id`、恢复默认 RoPE 初始化入口，并适配新版 Mask API。兼容层只在 Guard 加载进程内生效，不修改 `site-packages` 或 Hugging Face 模型缓存。

| 条件 | `target_only` | `speculative` | `draft_only` | `inline_guard` |
|---|---|---|---|---|
| `think_on` | 14B Target + posthoc Guard | vLLM Draft→Target + posthoc Guard | 0.6B Draft + posthoc Guard | vLLM Draft→Guard→Target |
| `think_off` | 14B Target + posthoc Guard | vLLM Draft→Target + posthoc Guard | 0.6B Draft + posthoc Guard | vLLM Draft→Guard→Target |

默认参数：100 条、base seed 42、每条样本 seed 为 `sha256(base_seed:sample_id) % (2**31-1)`、`max_new_tokens=4096`、K=5、Target `temperature=0.6/top_p=0.95/top_k=20`，Draft 使用 vLLM 的 greedy proposal，standard rejection sampling。两个 thinking 条件和四种模式对同一 `sample_id` 使用完全相同的 seed。

## 模型与单 GPU 约束

默认模型路径为：

```text
Draft:  /home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2
Target: /home/model_and_data/yxz/model/Huihui-Qwen3-14B-abliterated-v2
Guard:  /home/model_and_data/yxz/model/Qwen3Guard-Stream-4B
```

实际目录名不同时通过 CLI 或环境变量覆盖。父进程为每个隔离 worker 设置单个 `CUDA_VISIBLE_DEVICES`；worker 强制检查只能看到一张卡，Target、Draft、Guard 都使用逻辑 `cuda:0`，`tensor_parallel_size=1`。speculative worker 同时驻留三个模型。默认 `gpu_memory_utilization=0.72` 是为后加载的 4B Guard 留显存；若显存不足，应降低该值，而不是把 Guard 移到另一张卡。

## 在线插入点与不变性

`sdguard_sd.vllm_extension.SD4WorkerExtension` 通过 vLLM 的 `worker_extension_cls` 注入实际 GPU worker。对默认 V2 runner，它在运行时包装 vLLM 0.31 的 `PlainDraftModelSpeculator.propose()`；若该 0.31 构建显式使用 V1 runner，则回退包装 `GPUModelRunner.propose_draft_token_ids()`：

```text
Draft proposal 已返回 → Guard 增量观察/报警 → 原 Draft IDs 原样返回 → Target verification
```

适配层不修改 token tensor、不修改采样状态、不参与接受/拒绝，也不提前停止。每条 `inline_guard` 紧跟同 seed 的标准 `speculative` 请求；程序逐 token 比较最终 IDs，并比较 vLLM `detailed` per-request acceptance trace。任一不一致立即失败，因此不会错误复用人工标签。

按本次要求，SD_3 后补的离线 trace replay 不进入 SD_5；`inline_guard` 只做重新调用模型的在线生成与检测。

适配层只支持固定的 vLLM **0.31.0 base version**（允许类似 `0.31.0+cu129` 的本地构建后缀），且不修改 site-packages。升级 vLLM 时必须重新核对 hook 签名。

## 延迟口径

正式结果是 100 条、batch size 1、关闭 prefix cache、warm-up 和模型加载均排除的 wall latency：

- `target_only`：Target 模型生成、posthoc Guard、总 pipeline；
- `draft_only`：Draft 模型生成、posthoc Guard、总 pipeline；
- `speculative`：vLLM SD、posthoc Guard、总 pipeline；
- `inline_guard`：Guard prompt prefill、proposal、commit、Guard 总时延、含在线 Guard 的 pipeline，以及与同样本标准 SD 的 paired overhead；最终 posthoc Guard 只是审计，不计入 operational latency。

每项正式时延汇总 mean、median、P90、P95、min、max。SD 的 Draft proposal、Target verification、rejection sampling、scheduler/bookkeeping、Guard 各阶段 GPU kernel 时间来自额外 profiler 请求，写入 `vllm_profile_summary.json`；它们只用于归因诊断，不能当作 wall latency，也不混入正式 100 条统计。

## 运行

在装有对应 CUDA vLLM wheel 的 Linux 环境中：

CUDA 13.0 服务器的已验证版本、自动安装脚本、环境校验和跨服务器复用说明见
[`INSTALL_CUDA130.md`](INSTALL_CUDA130.md)。不要使用旧的 Transformers 4.57.6
环境运行 SD_5；当前 vLLM 0.31.0 环境固定 Transformers 5.17.0。

```bash
cd /root/autodl-tmp/Code/SD_Guard/SD_5

GPU_ID=0 \
OUTPUT_DIR=/root/autodl-tmp/Code/SD_Guard/SD_5/outputs/sd5_2x4_100 \
bash scripts/run_matrix.sh --local-files-only
```

也可直接运行：

```bash
python run_matrix.py --output-dir /path/to/output --gpu-id 0 --local-files-only
```

一条命令依次启动 Target、Draft、speculative 三个隔离进程，避免不同 vLLM engine 的 CUDA allocator/KV cache 相互污染；这不是三张 GPU，所有进程串行复用同一物理 GPU。`speculative` 与 `inline_guard` 在同一个 engine/Guard 实例中成对运行。

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
  raw/vllm_profile_summary.json       # 两种 SD cell
  blind/blind_outputs.jsonl
  private/original_dataset_audit.jsonl
<matrix>/analysis/matrix_summary.json
<matrix>/analysis/matrix_guard_summary.csv
<matrix>/analysis/latency_report.json
```

生成完成后按 `JUDGING_GUIDE.md` 给六个非 inline cell 做盲评，并把 `judge_labels.jsonl` 放进对应 `blind/` 目录。随后运行：

```bash
MATRIX_DIR=/path/to/output bash scripts/analyze_matrix.sh
```

脚本会再次逐 token 核验两组 `speculative`/`inline_guard` 输出及保存的 acceptance-trace 等价证明，通过后才复制两份 speculative 标签给 inline cell；然后统一计算八组 accuracy、recall、FNR、FPR、F1 并重建矩阵汇总。
