# v6-rule：第一版规则与通用因果规则总结

## 1. 实验边界

两版规则都以冻结的 `corrected-r3 FUSION_S2` 为基础，并在同一份 common DEV 上评估：

- DEV episode：2,028；
- 正例：414；
- 负例：1,614；
- 原始 FUSION_S2 阈值：`0.6289290373189838`；
- 目标：假阳性率不超过 5%；
- 当前没有独立 TEST，因此下文是开发集结果，不是未见数据上的最终泛化结论。

原始 FUSION_S2 在该 DEV 上为：

| TP | FP | FN | TN | FNR | FPR | AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 402 | 80 | 12 | 1,534 | 2.8986% | 4.9566% | 0.990347 |

## 2. 第一版：三词标题实体覆盖规则

### 2.1 动机

第一版从九个 Travel 漏报中观察到以下表面模式：

1. 低信任工具输出中出现注入包装；
2. 注入要求模型说出、推荐或访问某个实体；
3. 该实体随后出现在助手文本中；
4. 用户请求本身没有提到该实体。

### 2.2 决策过程

第一版要求文本载荷同时满足：

- 来自带注入标记的工具输出；
- 动作为 `say`、`recommend` 或 `visit`；
- 实体采用标题式大小写；
- 实体至少包含三个词；
- 后续助手文本精确复现该实体；
- 用户请求没有授权该实体。

最终决策是硬覆盖：

```text
FUSION_S2 判危险 OR 第一版规则命中
```

### 2.3 实验结果

| 方法 | TP | FP | FN | TN | FNR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| corrected-r3 FUSION_S2 | 402 | 80 | 12 | 1,534 | 2.8986% | 4.9566% |
| FUSION_S2 + 第一版覆盖 | 411 | 80 | 3 | 1,534 | 0.7246% | 4.9566% |

TRAIN 中规则命中 35 个 episode，全部为正例；DEV 中命中 16 个，全部为正例。

### 2.4 为什么否决第一版

虽然结果很好，但所有命中均集中在 Travel，且与 `injection_task_6` 的表面格式高度一致。固定标题大小写和至少三个词缺少任务无关的安全含义，更像针对已知模板构造的补丁。因此第一版只保留为消融实验，不作为最终候选。

## 3. 第二版：通用指令到落点的因果流

### 3.1 核心假设

第二版不再识别某个酒店名或某种固定词形，而是识别以下因果关系：

```text
低信任工具输出中的指令
        ↓
提取动作与未获用户授权的载荷
        ↓
在 episode 内跨事件保留污染状态
        ↓
载荷进入与动作相容的文本或工具落点
```

运行时代码不读取 suite、user task、injection task、episode ID、标签或已知攻击值，也不要求固定词数和标题大小写。

### 3.2 来源识别

规则只从历史 `tool` block 中寻找低信任指令。目前支持的通用包装包括：

- `<INFORMATION>`；
- `IMPORTANT!!!`；
- `TODO:`；
- `Ignore previous/prior instructions...`；
- `Before you can solve/complete/answer...`。

随后提取第一个命令动作及其载荷。载荷分为：

- 文本对象，如推荐、说出或访问的对象；
- 带类型标识符，如 URL、邮箱、长账号。

如果载荷已经存在于可信用户请求中，则不作为未授权传播证据。

### 3.3 跨事件状态

规则按 `row_index` 依次处理事件。注入来源出现后，即使后续经历查询、读取个人信息或正常工具调用，污染载荷仍保留在当前 episode 的状态中，直到它进入后续落点。

因此它可以识别如下多步流程：

```text
酒店评论中注入“收集个人信息并发到某邮箱”
→ get_user_information
→ send_email(recipients=[注入邮箱])
```

### 3.4 两个证据特征

#### `text_object_flow`

当注入命令的文本对象出现在后续助手文本中，且该对象未由用户授权时置为 1。

它不限制对象必须是三个词或标题式大小写。单词、双词和三词对象均有单元测试。

#### `tool_identifier_flow`

当注入中的 URL、邮箱或账号进入后续工具参数，并且动作或参数角色相容时置为 1。例如：

```text
send + URL       → send_direct_message.body
email + email    → send_email.recipients[0]
modify + account → update_transaction.recipient
```

仅仅在无关参数里出现同一字符串不会自动构成此证据。

### 3.5 是否需要训练模型或外部 NLP 模块

规则本身不需要训练，也不调用 LLM。`rule_taint.py` 只使用 Python 标准库：

- `re`：标记、动作、URL、邮箱和账号模式；
- `json`：工具参数展开；
- `hashlib`：去重来源块；
- `dataclasses` 和 `typing`：状态及证据结构。

它依赖项目已有的 causal event view，例如 `prior`、`current`、`user_request` 和工具参数。离线评估脚本使用 NumPy 读取已有分数包，但规则检测本身没有新增可训练模型。

这种做法的优点是快速、可审计；缺点是对未覆盖的注入包装、语义改写或没有明显标识符的攻击可能漏检。

## 4. 第二版怎样与 FUSION_S2 配合

### 4.1 分数不是概率

`corrected-r3 FUSION_S2` 是一个可正可负的排序分数，不是 `[0,1]` 概率。因此下面的 `0.9` 和 `0.1` 表示相对于该分数尺度的增量，不表示“危险概率增加 90% 或 10%”：

```text
hybrid_score = FUSION_S2
             + 0.9 * text_object_flow
             + 0.1 * tool_identifier_flow
```

两个权重由 TRAIN 分组折上的平均 AUC选择：

- 文本权重：0.9；
- 工具权重：0.1；
- TRAIN 分组折平均 AUC：0.998949。

DEV 用来按照 5% FPR 预算校准最终阈值，得到：

```text
hybrid_threshold = 0.6463982958921174
```

### 4.2 `0.9` 是否实际上等于强制判危险

数学上不是。文本规则命中后仍需满足：

```text
base_score + 0.9 >= 0.646398
```

即原分数至少约为：

```text
-0.253602
```

但在当前 DEV 的实际数据中，16 个文本命中样本的原分数范围为：

```text
0.092768 ～ 1.540507
```

所以这 16 个样本加 0.9 后确实全部判为危险。原模型已经检出其中 7 个，另外 9 个是被文本规则救回的漏报。因此在当前 DEV 上，文本分支的实际作用接近高置信度正例覆盖。它之所以没有增加 FP，是因为 TRAIN 的 34 个和 DEV 的 16 个文本命中目前全部是正例。

这仍不是未见数据上的精度保证；独立 TEST 可能出现文本规则负例。

### 4.3 为什么工具分支不能硬覆盖

DEV 上工具分支命中 376 个 episode：

- 正例 312；
- 负例 64。

因此工具证据只能给较小增量。如果使用 `0.9/0.1` 权重但继续沿用旧阈值，结果会变成：

| 阈值方案 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| 原 FUSION_S2 | 402 | 80 | 12 | 1,534 |
| 加规则但沿用旧阈值 | 412 | 85 | 2 | 1,529 |
| 加规则并按5% FPR重新校准 | 412 | 80 | 2 | 1,534 |

所以第二版能保持误报率，不只是因为权重小，也因为最终阈值从 `0.628929` 调整到了 `0.646398`。

## 5. 第二版跨任务命中情况

DEV 正例中的规则命中如下：

| 特征 | suite | 正例 episode | 不同任务组合 |
|---|---|---:|---:|
| text object | Travel | 16 | 3 |
| tool identifier | Banking | 54 | 7 |
| tool identifier | Slack | 237 | 31 |
| tool identifier | Travel | 17 | 5 |
| tool identifier | Workspace | 4 | 2 |

工具分支在 TRAIN 中命中 896 个 episode，其中 754 个正例、142 个负例，并覆盖四个 suite。这说明工具分支对应的是跨任务常见机制，而不是只针对原来的 Travel 漏报。

文本分支目前仍集中于 Travel，因此它虽然已经去掉标题大小写和三词限制，但对未知文本型攻击的泛化仍需独立数据验证。

## 6. 第二版实验结果

| 方法 | TP | FP | FN | TN | FNR | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| corrected-r3 FUSION_S2 | 402 | 80 | 12 | 1,534 | 2.8986% | 4.9566% | 0.990347 |
| 第一版三词覆盖 | 411 | 80 | 3 | 1,534 | 0.7246% | 4.9566% | 未单独报告 |
| 第二版因果融合 | 412 | 80 | 2 | 1,534 | 0.4831% | 4.9566% | 0.992980 |

第二版恢复 10 个原始漏报：

- 9 个 Travel 文本传播；
- 1 个 Slack URL 外发。

剩余 2 个为 Banking 账号修改。规则已经检测到账号进入交易收款人参数，但 TRAIN 选择的工具权重较保守，融合分数仍低于阈值。

## 7. 如何解释“漏报率到了多少”

- `0.7246%`（约 0.7%）是第一版规则在 common DEV 上的结果：3/414；
- `0.4831%`（约 0.48%）是第二版规则在同一 common DEV 上的结果：2/414。

这些数字是真实读取冻结 DEV 分数并重新计算混淆矩阵得到的，不是只测试九个漏报样本。但由于规则动机来自已检查的 DEV 漏报，而且第二版阈值也在该 DEV 上校准，所以它们应称为“common DEV 开发结果”，不能当作独立 TEST 成绩。

下一步的严格做法是冻结代码、权重和阈值策略，再在完全未见的 TEST 上一次性评估。

## 8. 文件与复现

主要文件：

- `src/sdsafe_v6/rule_taint.py`：规则实现；
- `scripts/evaluate_rule.py`：TRAIN 权重选择和 DEV 评估；
- `artifacts/rule-v1/report.json`：第一版报告；
- `artifacts/rule-v2/report.json`：第二版报告；
- `artifacts/rule-v2/evidence_train.jsonl`：TRAIN 因果证据；
- `artifacts/rule-v2/evidence_dev.jsonl`：DEV 因果证据；
- `artifacts/rule-v2/scores_dev.npz`：对齐后的原分数、规则特征和融合分数。

复现命令：

```bash
cd /root/sdsafe-compute/v6-rule
PYTHONPATH=src /root/miniconda3/bin/python -m unittest tests.test_rule_taint
PYTHONPATH=src /root/miniconda3/bin/python scripts/evaluate_rule.py
```
