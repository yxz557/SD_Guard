# SD_latency_test

这是从 `Code/SD_0` 单独摘出的 Draft/Target 测试工具，不加载数据集或 Guard。目标是回答三个问题：

1. Target-only 与 SD 的输出差别多大；
2. 去掉过程记录后，SD 本身是否比 Target-only 快；
3. 保留 SD round trace 时，Draft、Target verify/接受修正、trace 搬运等阶段分别耗时多少。

## 保留的 SD_0 策略

- Transformers `4.57.6` 原生 assisted generation；
- Draft：Qwen3-0.6B；Target：Qwen3-4B；
- `num_assistant_tokens=5`；
- `num_assistant_tokens_schedule=constant`；
- `assistant_confidence_threshold=0.4`；
- sampling 参数为 `temperature=0.6, top_p=0.95, top_k=20`；
- greedy/sample 参数会直接传给 `generate()`，并禁用模型默认值回填，避免
  Qwen 自带的 `do_sample=True` 把 greedy 覆盖成 sampling；
- trace 仍记录每轮 proposed、accepted、committed token 和 commit kind。

已有 `results/SD_0/...` 的旧结果元数据记录的是 `assistant_confidence_threshold=0.0`。本目录默认按当前 `Code/SD_0` 源码使用 `0.4`；如需严格复现旧结果，传入：

```bash
--assistant-confidence-threshold 0
```

Draft 和 Target 默认都放在逻辑 `cuda:0`，启动脚本也只暴露一张 GPU。

## 为什么同时测 greedy 和 sampling

- **greedy**：用于逐 token 输出对比。在理想精度下 Target-only 和 lossless SD 应一致；BF16 中块状 verify 与逐 token forward 可能在临界 logits 上产生差异，所以报告也保存首个不同 token、公共前缀长度和文本相似度。
- **sampling**：复现 SD_0 主实验的采样路径。它用于性能观察，但同 seed 下两种算法消耗随机数的顺序可能不同，因此不要求单条输出逐 token 相同。

默认 `--benchmark-mode both` 会同时测试两种模式。只想快速检查时可用 `--benchmark-mode greedy`。

## 运行

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/Code/SD_latency_test
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

BENCHMARK_WARMUP=2 BENCHMARK_REPEATS=10 \
bash scripts/run_latency_test.sh --local-files-only
```

直接调用入口也可以：

```bash
python run_latency_test.py \
  --draft-model /home/model_and_data/yxz/model/Qwen3-0.6B \
  --target-model /home/model_and_data/yxz/model/Qwen3-4B \
  --device cuda:0 \
  --benchmark-mode both \
  --benchmark-warmup 2 \
  --benchmark-repeats 10 \
  --max-new-tokens 256 \
  --run-name baseline-256 \
  --local-files-only
```

只检查 greedy 是否还会被 sampling 覆盖时，不必运行完整延迟和 trace。下面的
快速脚本只做四次短生成，并故意使用更明显的 sampling 参数；因此它只用于模式
切换检查，不用于延迟比较：

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/Code/SD_latency_test
CUDA_VISIBLE_DEVICES=0 python check_greedy_vs_sampling.py --local-files-only
```

结果默认写入仓库的
`SD_latency_test/outputs/<时间>_mode-check_seed<seed>_tok<tokens>/`。终端显示
`PASS` 表示 Target-only 和 SD 的 greedy 输出都与各自的 sampling 输出不同；
同时可在 `output_comparison.txt` 中查看四行相邻输出。

可重复传入 `--prompt`。也可以用 `--prompts-file` 读取 UTF-8 文本（每行一个 prompt）或 JSONL：

```json
{"id":"p1","prompt":"请解释天空为什么是蓝色。"}
```

## 输出

两个入口默认都会在仓库根目录的 `SD_latency_test/outputs/` 下为每次实验创建
独立目录，不会覆盖上一轮。例如：

```text
/home/model_and_data/yxz/Code/SD_Guard/SD_latency_test/outputs/20261014_153045_both_seed12345_tok96/
```

可以用 `--run-name` 给本轮结果增加容易辨认的名称：

```bash
--run-name no-think-greedy
```

对应目录类似：

```text
/home/model_and_data/yxz/Code/SD_Guard/SD_latency_test/outputs/20261014_153045_no-think-greedy/
```

如需写入完全指定的目录，使用：

```bash
--output-dir /path/to/exact/output-directory
```

启动脚本也支持环境变量 `OUTPUT_DIR=/path/to/exact/output-directory`。只有显式指定路径时才会复用该目录；默认行为始终是一轮一个新目录。

每个运行目录包含：

- `output_pairs.txt`：每个 prompt 的 `TARGET_ONLY` 与 `SD` 输出严格相邻。换行被转义为 `\n`，因此每个输出只占一个物理行，便于人工逐行比较。
- `latency_report.json`：完整配置、请求及实际启用的 Attention 后端、运行环境、
  每次延迟、speedup、输出差异、trace 校验和阶段汇总。
- `sd_round_trace.jsonl`：每个 speculative round 的 token 和阶段计时。

`speedup > 1` 表示 SD 更快。

## 三条延迟路径

报告有意区分：

1. `target_only`：Target 原生生成；
2. `sd_untraced`：原生 assisted generation，不安装 trace，代表 SD 算法本身的主要性能；
3. `traced_phase_profile`：保留 round trace 并测量阶段，用于定位和量化记录开销。

阶段字段：

- `draft_proposal_ms`：Draft 生成候选；
- `verify_and_decide_ms`：Target verify、logits 处理、接受/拒绝和修正采样的组合时间；
- `target_forward_cuda_ms`：上一个字段中仅 Target forward 的 CUDA event 时间，是其子集，不能再次相加；
- `proposed_trace_ms` / `result_trace_ms`：候选与提交 token 搬到 CPU 并转换成 Python 对象；
- `strategy_update_ms`：Transformers candidate strategy 更新；
- `generate_framework_other_estimate_ms`：未落入上述 round hook 的 generate 框架开销估算；
- `trace_slowdown_ratio`：单次 traced SD 相对 untraced SD 中位延迟的倍率。

阶段 profile 会在边界主动同步 CUDA，以获得可解释的阶段时间，因此它是诊断路径，不应作为生产吞吐。最终判断 SD 是否加速，应看 warm-up 后多轮 `sd_untraced` 与 `target_only` 的中位数和聚合 tokens/s。

## 建议的正式测法

- `warmup >= 2`，`repeats >= 10`；
- prompt 不要只用一条，至少覆盖短输出、长 thinking、代码和自然语言；
- 同时看延迟与 tokens/s，避免 EOS 导致输出长度不同而误判；
- 第一次固定当前策略；确认瓶颈后，再分别 sweep `num_assistant_tokens` 和 confidence threshold，避免一次改变多个变量。

## vLLM `draft_model` 对照

`run_vllm_latency_test.py` 是与上述 Transformers 路径并列的 vLLM
benchmark。它不会改写或调用 Transformers 的 `assistant_model` 实现，而是使用
vLLM 官方 `speculative_config={"method": "draft_model", ...}` 路径。

### 独立环境

vLLM 会约束自己的 PyTorch、Transformers 和 CUDA wheel。不要把它安装到
`requirements.txt` 对应的 Transformers 4.57.6 环境中。`vllm==0.31.0` 要求
Python `>=3.10,<3.15`。必须显式选择 Python 3.10 或 3.11；不要直接使用指向未知
版本的 `python -m venv`。

如果服务器安装了 Python 3.10：

```bash
python3.10 --version
python3.10 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
```

如果服务器依赖 Conda：

```bash
conda create -n sd-vllm python=3.10 -y
conda activate sd-vllm
python -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
```

安装前输出的 Python 必须是 `3.10.x`、`3.11.x`、`3.12.x`、`3.13.x` 或
`3.14.x`。如果 pip 输出大量 `Requires-Python >=3.10` 并忽略新版 vLLM，说明当前
解释器版本不符合要求；不要因此退回 vLLM 0.11.0，因为本 harness 使用了新版
`draft_sample_method` 和 `LLM.get_metrics()` 接口。

随后再根据该机器的 NVIDIA driver 选择 vLLM 官方 wheel。vLLM 0.31.0 的 PyPI
默认 wheel 使用 CUDA 13.0；如果 `nvidia-smi` 显示驱动最多支持 CUDA 12.9，直接执行
`pip install -r requirements-vllm.txt` 会装入不兼容的 CUDA 13.0 PyTorch。此时使用
官方 release commit 对应的 CUDA 12.9 wheel：

```bash
python -m pip install --upgrade uv
uv pip install vllm==0.31.0 --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
```

安装后必须检查实际解析结果：

```bash
python -c 'import torch, vllm; print("vLLM", vllm.__version__); print("Torch", torch.__version__); print("Torch CUDA", torch.version.cuda); print("CUDA available", torch.cuda.is_available())'
```

CUDA 12.9 驱动上，`Torch CUDA` 应输出 `12.9`，且 `CUDA available` 应为 `True`。
由于 PyTorch/vLLM CUDA 二进制必须成套匹配，若已经装过默认 CUDA 13.0 包，推荐创建
新的 Conda 环境后执行上面的 `uv` 命令，不要在旧环境中逐个覆盖依赖。

当前 harness 固定 `vllm==0.31.0`，并把实际 vLLM、PyTorch、CUDA 和 GPU 信息写入
结果。若服务器 driver 不支持该版本的官方 wheel，应先选择一个与 driver 匹配且
支持 Qwen3 `draft_model` 的 vLLM 版本，再同时修改 `requirements-vllm.txt` 和
`--required-vllm-version`；不要无记录地更换版本。

### 官方固定 K 策略

默认配置对应 vLLM 的普通小模型 Draft 路线：

- Target：Qwen3-4B；
- Draft：Qwen3-0.6B；
- `method=draft_model`；
- `num_speculative_tokens=5`；
- `draft_sample_method=greedy`；
- `rejection_sample_method=standard`；
- BF16、TP=1、batch=1；
- `max_model_len=2048`、输出上限 256；
- prefix caching 关闭，避免重复 prompt 在后续 repeat 中命中旧 KV；
- `enforce_eager=false`，保留 Target runner 的 CUDA Graph 路径；vLLM 0.31.0 的
  `PlainDraftModelSpeculator` 仍明确不支持 CUDA Graph，profiling 报告会核验并记录
  这一实现事实；
- attention backend 由 vLLM 自动选择，不传 Transformers 的
  `attn_implementation`。

运行：

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/Code/SD_latency_test
CUDA_VISIBLE_DEVICES=0 \
BENCHMARK_WARMUP=2 BENCHMARK_REPEATS=10 \
bash scripts/run_vllm_latency_test.sh --local-files-only
```

或直接调用：

```bash
python run_vllm_latency_test.py \
  --draft-model /home/model_and_data/yxz/model/Qwen3-0.6B \
  --target-model /home/model_and_data/yxz/model/Qwen3-4B \
  --benchmark-mode greedy \
  --benchmark-warmup 2 \
  --benchmark-repeats 10 \
  --batch-size 1 \
  --max-new-tokens 256 \
  --num-speculative-tokens 5 \
  --draft-sample-method greedy \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --no-enforce-eager \
  --profile-vllm \
  --profile-max-new-tokens 64 \
  --local-files-only
```

`--num-assistant-tokens` 是 `--num-speculative-tokens` 的兼容别名。HF 的
`assistant_confidence_threshold` 和 `assistant_schedule` 在这条固定 K 的 vLLM
路线中没有直接对应参数，因此不会传入。

如需 sampling，可使用：

```bash
--benchmark-mode sample \
--temperature 0.6 \
--top-p 0.95 \
--top-k 20
```

官方普通 Draft 配置默认使用 greedy drafter。`--draft-sample-method probabilistic`
仅用于额外比较更接近当前 Transformers sampling Draft 的行为，不应与默认公开配置
混为同一组结果。

### 为什么使用两个 worker

Target-only engine 和 SD engine 不能通过给同一个 `LLM` 对象临时增删 Draft 来切换。
每个 engine 都会管理模型权重、KV cache、调度器和 CUDA Graph。因此父进程按顺序
启动两个全新的子进程：

```text
Target-only worker -> 退出并释放 GPU -> speculative worker
```

默认顺序是 `--engine-order target-first`；可用 `--engine-order sd-first` 做反向顺序
复核。模型加载时间单独记录，不计入请求 latency。

### batch

`--batch-size` 表示一次 `llm.generate()` 中的不同请求数量。输入 prompt 数量必须能被
batch size 整除，以免最后一个不完整 batch 污染中位延迟。batch 大于 1 时应通过
`--prompts-file` 提供足够多且不同的 prompt，例如：

```bash
python run_vllm_latency_test.py \
  --prompts-file prompts.jsonl \
  --batch-size 4 \
  --run-name vllm-batch4 \
  --local-files-only
```

报告中的单次 `total_ms` 是整个 batch 的阻塞式 `llm.generate()` 延迟；吞吐为该
batch 所有输出 token 数除以 batch 时间。

### 定位 vLLM SD 内部瓶颈

加 `--profile-vllm` 后，脚本先完整执行不带插桩的正式测速，再在 speculative worker
退出前依次执行一个同形状的无 profiler 基线请求和一个 PyTorch profiler 请求。诊断
请求默认最多生成 64 tokens；可用 `--profile-max-new-tokens` 调整。两者都不进入
latency/speedup 的统计。

使用封装脚本时也可以通过环境变量开启：

```bash
PROFILE_VLLM=1 PROFILE_MAX_NEW_TOKENS=64 \
bash scripts/run_vllm_latency_test.sh --local-files-only
```

`vllm_profile_summary.json` 中重点查看：

- `diagnosis.signals`：把接受率、普通 Draft CUDA Graph 支持情况和可归因的主导 GPU
  阶段汇总成直接可读的诊断信号；
- `gpu_kernel_time_breakdown.stages`：Draft proposal、Target verification、rejection
  sampling、调度/记账和未归类部分的 GPU kernel 时间及占比；
- `gpu_kernel_time_breakdown.attribution_coverage`：能可靠归类的 GPU 时间比例。覆盖率
  低时不要直接拿阶段百分比下结论，应在 Perfetto 中打开原始 trace；
- `implementation_findings`：运行时检查当前安装版本的普通 Draft speculator 是否支持
  CUDA Graph；
- `profiler_overhead_ratio`：profiler 请求相对同 batch、同 token cap 无插桩诊断请求的
  倍率，只用于判断 profiler 扰动，不能作为性能结果。

阶段百分比按 GPU kernel 时间计算，不是同步后的端到端 wall time。原始 trace 和
vLLM 自带的 operator 表保存在 `vllm_torch_profile/`，可以用
<https://ui.perfetto.dev/> 打开 `*.pt.trace.json.gz` 继续检查。

### vLLM 输出

每轮结果目录包含：

- `latency_report.json`：Target-only/SD 延迟、吞吐、speedup、输出比较和环境；
- `output_pairs.txt`：相邻的 Target-only 与 SD 文本；
- `vllm_spec_decode_metrics.json`：去掉 warm-up 后的 Draft/接受统计；
- `vllm_profile_summary.json`：开启 `--profile-vllm` 时生成的瓶颈摘要；
- `vllm_torch_profile/`：开启 profiling 时生成的原始 Chrome trace 和 operator 表；
- `vllm_target_worker.json`、`vllm_sd_worker.json`：两个独立 engine 的原始结果；
- 两个 `*_worker_config.json`：实际传给 worker 的完整配置。

vLLM 公共指标可以提供 speculative step 数、proposed/accepted token 数、平均接受
长度和逐位置接受率。profiling 模式额外从官方 PyTorch trace 归因 GPU kernel 时间；
它不等价于当前 Transformers tracer 的逐轮同步阶段 wall time，因此 vLLM 路径仍
不会生成 `sd_round_trace.jsonl`。

最终 SD 加速必须比较同一后端内的：

```text
vLLM Target-only vs vLLM Target + Draft
```

Transformers Target-only 与 vLLM SD 的差值同时包含推理引擎变化，不能全部归因于
speculative decoding。
