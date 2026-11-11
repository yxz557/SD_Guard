# Guard Training 实验 01：结果与分析

## 1. 结果范围

本文分析：

```text
Qwen3-4B final LayerNorm trace_last hidden state
  + LayerNorm/MLP classification head
  + inverse positive-class weight
  + validation F2 threshold
  + seeds 42..51
```

方法和完整运行命令见：

[`experiment_01_trace_last_mlp_method.md`](experiment_01_trace_last_mlp_method.md)

远端聚合结果文件为：

```text
/root/autodl-tmp/Code/SD_Guard/results/formal/guard_training/
  qwen3_4b_task_injection_ctx40k_r1/experiments/
  trace_last_final_mlp_10seeds/aggregate_test_metrics.json
```

下列均值和标准差来自固定 test split 上 10 个训练随机种子。标准差只反映训练随机性，不代表重新划分数据后的不确定性。

## 2. 测试集组成

| 项目 | 数量 |
|---|---:|
| 测试样本 | 705 |
| 攻击成功正例 | 89 |
| 安全/攻击未成功负例 | 616 |
| 正例比例 | 12.62% |

正例定义为 `hijacked=true`。

## 3. 总体结果

| 指标 | 10 次均值 ± 样本标准差 | 解释 |
|---|---:|---|
| AP | **0.6858 ± 0.0138** | PR 排序质量；随机基准约为正例率 0.1262 |
| AUROC | **0.9399 ± 0.0043** | 随机正负样本对中，正例获得更高分数的概率约 94% |
| Accuracy | **84.11% ± 0.98%** | 所有样本中判断正确的比例 |
| Balanced Accuracy | **87.40% ± 0.53%** | 正类 Recall 与负类 Recall 的平均 |
| Positive Precision | **43.89% ± 1.56%** | 所有报警中真正攻击成功的比例 |
| Positive Recall | **91.80% ± 1.30%** | 成功攻击中被检出的比例 |
| Positive F1 | **59.36% ± 1.32%** | Precision 与 Recall 的调和平均 |
| Macro F1 | **74.74% ± 0.99%** | 正负两类 F1 等权平均 |
| 误报率 FPR | **17.00% ± 1.24%** | 安全 trace 中被误报为攻击的比例 |
| 漏报率 FNR | **8.20% ± 1.30%** | 成功攻击中未被检出的比例 |
| Validation F2 阈值 | **0.4606 ± 0.1708** | 每个 seed 单独在 validation 上选择 |

阈值范围为 `0.1164` 至 `0.7718`。AP 和 AUROC 很稳定，但阈值变化较大，说明排序能力稳定、分数刻度/概率校准不稳定。

### 3.1 平均混淆矩阵

10 次实验的平均计数为：

| 真实标签 | 预测安全 | 预测攻击 |
|---|---:|---:|
| 安全，共 616 条 | TN = 511.3 | FP = 104.7 |
| 攻击成功，共 89 条 | FN = 7.3 | TP = 81.7 |

因此：

```text
Accuracy  = (TP + TN) / 705              = 84.11%
Precision = TP / (TP + FP)               = 43.89%
FPR       = FP / (FP + TN)               = 17.00%
FNR       = FN / (FN + TP)               =  8.20%
Recall    = TP / (TP + FN) = 1 - FNR     = 91.80%
```

这里还要区分两种常被同时称为“误报”的量：

- FPR 为 17.00%：所有安全样本中有多少被误报；
- `1 - Precision` 为 56.11%：模型已经发出的报警中有多少是假警。

### 3.2 为什么 Accuracy 不应单独使用

测试集中负例占 87.38%。如果永远预测“安全”，Accuracy 可以达到：

```text
616 / 705 = 87.38%
```

但这种分类器会漏掉 100% 的攻击。因此实验 01 的 Accuracy 虽然低于全负预测基线，攻击 Recall 却从 0 提高到了 91.8%。本任务应主要结合 AP、AUROC、Balanced Accuracy、Precision、Recall、FPR 和 FNR 判断。

### 3.3 总体指标的初步含义

- AP 0.686 相比随机 AP 基准 0.126 高约 5.4 倍；
- AUROC 0.940 说明 hidden state 中存在明显的攻击成功排序信号；
- 10 个 seed 的 AP/AUROC 波动很小，结果不像偶然初始化造成；
- 当前 inverse class weight 和 F2 阈值偏向 Recall，因此漏报较低、误报和假警较多。

总体结果足以证明该方法是一个有效的 proof of concept，但不能单独证明它已经是稳定的统一 Guard。

## 4. 按 suite 分解

下表均为 10 个 seed 的平均值。Accuracy、FPR 和 FNR 根据各组平均正/负 Recall 计算。

| Suite | 样本/正例 | 正例率 | Accuracy | Balanced Acc. | Precision | Recall | FPR | FNR | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 110 / 25 | 22.73% | **86.73%** | **87.88%** | **65.07%** | 90.00% | 14.24% | 10.00% | 0.790 | 0.943 |
| `slack` | 80 / 37 | 46.25% | 51.75% | 55.02% | 48.95% | **98.65%** | **88.60%** | 1.35% | **0.831** | 0.828 |
| `travel` | 105 / 23 | 21.90% | 58.00% | 68.73% | 32.91% | 87.83% | **50.37%** | 12.17% | 0.436 | **0.746** |
| `workspace` | 410 / 4 | 0.98% | 96.41% | 79.62% | **16.30%** | 62.50% | 3.25% | **37.50%** | **0.198** | 0.967 |

### 4.1 Banking

Banking 是当前最均衡的 suite：

- AUROC 0.943；
- Precision 65.1%；
- Recall 90.0%；
- FPR 14.2%；
- FNR 10.0%。

这说明当前特征和 MLP 在 Banking 内部确实能够较好地区分攻击成功与未成功 trace。

### 4.2 Slack

Slack 的 Recall 达到 98.65%，但 FPR 达到 88.60%。模型在全局阈值下几乎把所有 Slack trace 都判为攻击：

- 排序指标仍有 AP 0.831、AUROC 0.828；
- 但阈值分类 Accuracy 只有 51.75%；
- 负类 Recall 只有 11.40%。

这更像明显的 suite/domain score shift：模型对 Slack 整体输出偏高风险分数。虽然没有显式输入 `suite_name`，Slack 的任务、工具名和 trace 风格仍能暴露领域。

### 4.3 Travel

Travel 的问题不只是阈值：

- AUROC 只有 0.746；
- AP 只有 0.436；
- FPR 达到 50.37%；
- Precision 只有 32.91%。

说明模型在 Travel 内部的实际排序区分能力也明显弱于 Banking。

### 4.4 Workspace

Workspace 的 410 条测试样本中只有 4 个正例：

- 一个正例判断变化就会让 Recall 改变 25 个百分点；
- 永远预测安全可获得 `406/410 = 99.02%` Accuracy；
- 当前 96.41% Accuracy 因而不能说明模型优秀；
- 当前只检出 62.5% 的攻击，Precision 只有 16.3%。

AUROC 0.967 与 AP 0.198 并不矛盾：极端类别不平衡时，AUROC 可以很高，而 AP 和实际报警 Precision 仍然很低。由于只有 4 个正例，Workspace 分组指标的不确定性很大。

### 4.5 Suite 等权结果暴露总体指标的乐观性

四个 suite 的 Balanced Accuracy 等权平均约为：

```text
(87.88% + 55.02% + 68.73% + 79.62%) / 4 = 72.81%
```

而将全部样本混合后的总体 Balanced Accuracy 是 87.40%。两者相差明显，说明 pooled overall 指标掩盖了 suite 间的不稳定性。

一个可能的机制是：

```text
Slack 样本整体分数较高，且 Slack 正例率高；
Workspace 样本整体分数较低，且 Workspace 正例率极低。
```

模型即使主要识别 suite，也可以在大量跨 suite 正负样本对上获得正确排序，从而推高总体 AUROC，却不一定能在每个 suite 内可靠地区分安全与攻击。

## 5. 按 attack type 分解

| Attack | 样本/正例 | 正例率 | Accuracy | Precision | Recall | FPR | FNR | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 141 / 8 | 5.67% | 83.76% | 25.14% | 93.75% | 16.84% | 6.25% | 0.543 | 0.954 |
| `ignore_previous` | 141 / 20 | 14.18% | **87.30%** | **53.03%** | 95.00% | **13.97%** | 5.00% | **0.841** | **0.965** |
| `important_instructions` | 141 / 21 | 14.89% | 80.78% | 43.00% | 88.10% | **20.50%** | 11.90% | 0.599 | 0.912 |
| `injecagent` | 141 / 16 | 11.35% | 84.26% | 41.91% | **100.00%** | 17.76% | **0.00%** | 0.641 | 0.941 |
| `tool_knowledge` | 141 / 24 | 17.02% | 84.47% | 52.86% | 86.25% | 15.90% | 13.75% | 0.758 | 0.932 |

按 attack 划分后的 AUROC 都在 0.91 以上，没有 suite 维度那么失控。主要差异包括：

- `ignore_previous` 综合表现最好；
- `injecagent` 在该测试集上没有漏掉正例，但存在较多误报；
- `important_instructions` 的 FPR 最高；
- `tool_knowledge` Precision 较好，但 FNR 相对高；
- `direct` 只有 8 个正例，5.67% 的低基率会显著压低 Precision，分组估计也不稳定。

因此，目前更明确的问题是跨 suite 稳定性，而不是所有 attack type 上都完全没有区分能力。

## 6. 阈值解释

分类头输出两个 logits，经 softmax 得到正类分数。对每个 seed：

```text
score >= validation threshold -> 预测攻击成功
score <  validation threshold -> 预测未攻击成功
```

阈值不是 MLP 的训练参数，而是训练后选择的 operating point：

- 降低阈值：Recall 上升、FNR 降低，但 FPR 上升、Precision 通常下降；
- 提高阈值：FPR 降低、Precision 通常上升，但 Recall 下降、FNR 上升。

实验 01 选择 validation F2 最大的阈值。10 个 seed 的阈值分别为：

```text
0.534, 0.116, 0.533, 0.329, 0.419,
0.571, 0.393, 0.772, 0.485, 0.453
```

不能把平均阈值 0.461 直接用于所有 checkpoint。每个 checkpoint 应使用自己保存的 validation 阈值；若构造 ensemble，应先聚合 validation 分数，再重新选择 ensemble 阈值。

由于采用类别加权且未做概率校准，softmax 分数首先应理解为风险排序分数，不能直接解释成真实攻击概率。

## 7. 结论

### 7.1 已经得到支持的结论

1. Qwen3-4B 最终层、最后 trace token 的 hidden state 中存在明显的攻击成功信号；
2. 一个很小的冻结特征 MLP 就能得到总体 AP 0.686、AUROC 0.940；
3. 10 个训练 seed 的排序指标稳定，结果不是单次初始化偶然现象；
4. 该方案适合作为后续特征和训练策略消融的第一版 baseline。

### 7.2 尚未得到支持的结论

1. 不能据此声称模型已经是稳定可部署的统一 Guard；
2. 不能据此声称模型主要学到的是危险行为，而不是 suite/task/工具先验；
3. 不能据此声称模型能泛化到训练中完全未见的 suite；
4. 不能据此声称模型能泛化到完全未见的 attack type；
5. 当前结果是 post-hoc 完整 trace 分类，不代表能在危险工具调用发生前阻止攻击。

### 7.3 最终定性

本实验是一个成功的 proof-of-concept baseline，但不是一个表现均匀的最终 Guard：

- pooled in-distribution 排序能力强；
- 当前 operating point 偏 Recall，整体误报率为 17%；
- suite 之间差异很大，Slack 与 Travel 暴露了明显问题；
- Workspace 正例太少，无法可靠判断；
- 总体 AUROC 0.94 很可能同时包含真实行为信号和 suite 先验，属于偏乐观的总体估计。

## 8. 后续对比实验优先级

### 8.1 不重新跑 Qwen3-4B 的特征消融

使用已经保存的 13GB 特征库和同一 split，依次比较：

1. `trace_last32_mean`, `layer=-1`；
2. `trace_mean`, `layer=-1`；
3. `readout_last`, `layer=-1`；
4. `trace_last` 的倒数第 2/4/8 层；
5. Linear head 与当前 MLP；
6. positive weight 为 `none`、`sqrt_inverse`、`4.0`、`inverse`。

### 8.2 检查模型究竟学到了什么

1. Task-only baseline：只输入任务，不输入行为 trace；
2. Leave-one-suite-out：完全拿掉一个 suite 训练，再测试该 suite；
3. Leave-one-attack-out：完全拿掉一种攻击训练，再测试该攻击；
4. suite-balanced sampling 或 loss weighting；
5. 对工具名、suite 特征进行 masking 的消融。

### 8.3 更贴近部署的指标

后续至少增加：

- Recall at FPR = 1%；
- Recall at FPR = 5%；
- Recall at FPR = 10%；
- 各 suite 的相同指标；
- suite macro average；
- worst-suite 指标；
- calibration curve、Brier score 或 ECE。

选择部署阈值时，建议先限定可接受 FPR，再在 validation 上最大化 Recall，而不是只使用 F2 最大阈值。

## 9. 公平比较规则

后续实验与实验 01 比较时必须：

- 使用相同的 `split_manifest.jsonl`；
- 使用相同 test 集；
- 使用同一组 10 个 seeds；
- 不根据 test 结果挑 seed；
- checkpoint 和阈值只能由 validation 决定；
- 同时报告 overall、by-suite、by-attack、macro 和 worst-group 指标；
- 明确区分训练随机性标准差和数据划分/采样不确定性。

