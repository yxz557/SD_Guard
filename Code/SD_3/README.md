# SD_3：Thinking × Generation Mode 的 2×4 Guard 对照实验

SD_3 在同一批 Qwen3GuardTest prompt 上控制变量，对比两个 thinking 条件与四种生成/检测模式：

| 条件 | `target_only` | `speculative` | `draft_only` | `inline_guard` |
|---|---|---|---|---|
| `think_on` | Target 直接生成 | Draft 提议、Target 验证，完成后 Guard | Draft 直接生成 | Draft 提议后立即 Guard，再由 Target 验证 |
| `think_off` | Target 直接生成 | Draft 提议、Target 验证，完成后 Guard | Draft 直接生成 | Draft 提议后立即 Guard，再由 Target 验证 |

`inline_guard` 是纯报警旁路：一旦 strict 的 `UU/CC` 或 loose 的 `UU` 出现，只锁存报警；不会截断、拒绝、替换 token，也不会提前结束生成。Target 仍按原 speculative sampling 流程接受、修正或补充 token。

同一个 inline cell 同时保存两个决策视图：

- `proposal_immediate`：每轮在 Target 验证前检查 `已提交前缀 C + Draft 提议 D1...Dk`。被 Target 拒绝的草稿也可能触发报警，因此能测量噪声。
- `verified_commit`：只检查最终提交的 token，包括被接受的 Draft token 和 Target correction/bonus token。它必须与完成生成后的原 `speculative` posthoc Guard 一致。

矩阵主比较把 `proposal_immediate` 当作第四种条件；`verified_commit` 是一致性对照，不另算第五种生成模式。

## 默认模型、GPU 和参数

```text
Draft:  /home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2
Target: /home/model_and_data/yxz/model/Huihui-Qwen3-8B-abliterated-v2
Guard:  /home/model_and_data/yxz/model/Qwen3Guard-Stream-4B
```

- Target 与 Draft：逻辑 `cuda:0`
- Guard：逻辑 `cuda:1`
- `sample_count=80`、`seed=42`
- 每条样本 seed：`sha256(base_seed:sample_id)`，跨八个 cell 对齐
- `max_new_tokens=4096`
- `temperature=0.6`、`top_p=0.95`、`top_k=20`
- `num_assistant_tokens=5`
- `assistant_confidence_threshold=0.4`
- `dtype=bfloat16`

在线 Guard 使用 KV cache。Target 给出验证结果后，cache 会裁到 `C + accepted Draft prefix`，复用已接受 token 的状态，并只增量写入 Target correction/bonus；被拒绝的草稿后缀不会泄漏到下一轮或 `verified_commit`。

## 给现有六组实验补两组：推荐顺序

现有结果目录：

```text
/home/model_and_data/yxz/Code/SD_Guard/SD_3/outputs/abliterated_2x3_80_4096
```

### 1. 先离线重放已有 speculative trace

先用少量样本验证环境、cache 回滚和统计口径：

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/SD_3

MATRIX_DIR=/home/model_and_data/yxz/Code/SD_Guard/SD_3/outputs/abliterated_2x3_80_4096 \
SAMPLE_LIMIT=5 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/replay_inline_guard.sh --local-files-only
```

pilot 成功后重放全部 80 条：

```bash
MATRIX_DIR=/home/model_and_data/yxz/Code/SD_Guard/SD_3/outputs/abliterated_2x3_80_4096 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/replay_inline_guard.sh --local-files-only --overwrite
```

离线 replay 只加载 Guard，不重新加载或运行 Draft/Target。它按保存的每轮 `proposed_ids / accepted_count / committed_ids` 重建两种报警逻辑，并严格校验 replay 后的 committed token 序列与原输出完全相同。增量 KV-cache forward 与原先一次性 full-forward 的 Guard token 标签可能因 bf16 数值边界出现少量差异，因此 replay 会记录每个差异 token、概率、标签迁移以及 strict/loose 最终决策是否一致，但不会因这种 Guard 计算路径差异中断实验。

### 2. 再跑两个真实在线 cell

```bash
cd /home/model_and_data/yxz/Code/SD_Guard/SD_3

MATRIX_DIR=/home/model_and_data/yxz/Code/SD_Guard/SD_3/outputs/abliterated_2x3_80_4096 \
SAMPLE_COUNT=80 \
MAX_NEW_TOKENS=4096 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_inline_matrix.sh --local-files-only
```

脚本只新增：

```text
<MATRIX_DIR>/think_on/inline_guard
<MATRIX_DIR>/think_off/inline_guard
```

每条在线样本都会执行两类硬校验：

1. 最终提交 token 与本次生成输出完全相同；
2. 本次 generated IDs 与对应的原 `speculative` cell 逐 token 完全相同。

此外会比较增量 `verified_commit` 与本次 posthoc Guard，完整记录 token 标签差异和 response 级 strict/loose 决策一致性；这些是实验结果而不是结构错误，因此不作为硬失败条件。

上述两项生成轨迹硬校验中的任何一项不一致都会立即报错，且不会复用人工标签。Guard 增量/一次性 forward 的分类差异只记录、不终止。全部 80 条的 generated IDs 与原 speculative cell 一致后，脚本才把原 speculative 的 `judge_labels.jsonl` 复制到 inline cell，运行 `proposal_immediate` 准确率分析并生成完整 2×4 汇总。

如果需要重跑已经存在的两个在线目录，明确追加 `--overwrite`。

## 输出与统计

每个 `inline_guard` cell 额外保存：

```text
raw/inline_guard_token_records.jsonl
raw/inline_guard_round_records.jsonl
raw/inline_guard_summary.json
```

`generation_records.jsonl` 的 `inline_guard` 字段包含：

- `proposal_immediate` 与 `verified_commit` 的 full-trace / answer-only、strict / loose 判定；
- proposal-only 报警数；
- 提议触发在 Target 验证后存活或被拒绝的轮数；
- Safe / Controversial / Unsafe 草稿 token 各自的接受率；
- Guard prompt prefill、proposal、commit 与总在线延迟；
- 相对原 speculative 运行的观测生成延迟差；
- generated-ID 与 posthoc 一致性结果。

最终矩阵汇总写入：

```text
<MATRIX_DIR>/analysis/matrix_summary.json
<MATRIX_DIR>/analysis/matrix_guard_summary.csv
```

汇总器会校验八个 cell 的样本、顺序和受控参数，并再次断言每个 thinking 条件下 `speculative` 与 `inline_guard` 的 generated IDs 完全相同。

## 从零运行原六组

`scripts/run_matrix.sh` 仍负责原始 2×3 六组，便于与已经完成的实验保持兼容：

```bash
SAMPLE_COUNT=80 \
OUTPUT_ROOT=/path/to/new_matrix \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_matrix.sh --local-files-only
```

六组完成后，把该目录作为 `MATRIX_DIR` 依次运行上面的 replay 和 `run_inline_matrix.sh`，即可扩展为 2×4。

`max_new_tokens=4096` 由 thinking 和 final answer 共同使用。达到上限但未 EOS 的样本标记为 `truncated=true`；Guard 不会静默裁剪超长上下文。
