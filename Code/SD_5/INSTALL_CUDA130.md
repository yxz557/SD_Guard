# SD_5：CUDA 13.0 / Transformers 5.17 环境安装与复用清单

本文档对应 SD_5 的单 GPU vLLM 0.31.0 + Transformers 5.17 路径。已验证环境快照位于仓库根目录：

- `sd4-env-summary.txt`：关键运行时和 GPU 信息；
- `pip-check.txt`：Python 依赖一致性检查；
- `requirements-sd4-cu130.lock.txt`：完整环境快照，仅用于审计和版本比对。

完整 freeze 中含有 AutoDL 基础环境附加包，且 `torch==2.13.0` 这一文本本身不携带
`+cu130` wheel 来源信息。因此不要把该文件脱离安装命令直接当作跨服务器的唯一
安装入口。正式安装使用 `requirements-vllm.txt` 和显式的
`--torch-backend=cu130`。

## 1. 已验证的核心组合

| 项目 | 版本/配置 |
|---|---|
| OS | Linux x86_64，glibc 2.35 |
| Python | 3.10.22 |
| vLLM | 0.31.0 |
| PyTorch | 2.13.0+cu130 |
| Torch CUDA runtime | 13.0 |
| Transformers | 5.17.0 |
| Tokenizers | 0.23.3 |
| Datasets | 4.8.5 |
| Accelerate | 1.15.0 |
| Triton | 3.7.1 |
| FlashInfer | 0.7.0.post1 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition，SM 12.0，94.97 GiB |
| 独立 flash-attn/xformers | 不安装 |

当前 `pip check` 为 `No broken requirements found.`。这证明包依赖一致，但不等于
Target、Draft、Guard 和 vLLM hook 已经完成端到端验证；正式实验前仍必须执行第 5 节
的冒烟测试。

## 2. 新服务器前置检查

要求：

1. Linux x86_64；推荐 Ubuntu 22.04 或等价的 glibc >= 2.28 系统；
2. NVIDIA 驱动 R580 或更新版本；`nvidia-smi` 右上角应显示 CUDA 13.0 或更高；
3. vLLM 支持的 NVIDIA GPU，compute capability >= 7.5；
4. SD_5 默认 14B Target + 0.6B Draft + 4B Guard 同卡，建议约 80 GiB 以上显存；
5. 只暴露一张 GPU，SD_5 固定 `tensor_parallel_size=1` 和 batch size 1；
6. 环境、模型和输出应预留足够磁盘空间。无需另外安装系统 CUDA Toolkit 或 nvcc。

先检查：

```bash
nvidia-smi
uname -m
ldd --version | head -n 1
```

CUDA 13.0 GA 对应 Linux 驱动至少为 580.65.06；在托管服务器上不要自行覆盖宿主机
驱动，改选满足要求的实例镜像或物理主机。

## 3. 推荐安装：Conda + uv + PyPI CUDA 13.0 wheel

以下示例把环境与 uv 缓存放在 AutoDL 数据盘的同一文件系统，避免跨文件系统复制：

```bash
source /root/miniconda3/etc/profile.d/conda.sh

conda create \
  -p /root/autodl-tmp/conda-envs/sd4-vllm-cu130 \
  python=3.10.22 \
  -y

conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
```

进入 SD_5 后执行自动安装脚本：

```bash
cd /root/autodl-tmp/Code/SD_Guard/SD_5
bash scripts/install_cuda130.sh
```

该脚本显式使用 `--torch-backend=cu130`，从默认发布源安装 vLLM 0.31.0，不依赖
`wheels.vllm.ai` 的 cu129 commit wheel。它还会运行 `pip check` 和 CUDA/GPU 校验。

等价的手动命令为：

```bash
python -m pip install --upgrade pip
python -m pip install "uv==0.13.1"

uv pip install \
  --python "$(command -v python)" \
  --torch-backend=cu130 \
  -r requirements-vllm.txt

python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/check_cuda130_env.py
```

不要追加 cu129 的 `wheels.vllm.ai/<commit>` URL，不要再单独安装另一个 Torch wheel，
也不要为了运行 SD_5 额外安装 `flash-attn` 或 `xformers`。

## 4. 安装后必须核验

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_cuda130_env.py
python -m pip check
python -m unittest discover -s tests -v
```

校验脚本要求看到：

- Python 3.10.x；
- vLLM 0.31.0；
- Torch 2.13.0，`torch.version.cuda == "13.0"`；
- Transformers 5.17.0；
- 一张可见 GPU；
- 驱动主版本至少为 580；
- compute capability 至少为 7.5。

## 5. 一条样本的端到端冒烟测试

根据新服务器实际目录设置路径。不要沿用脚本中的旧服务器默认路径：

```bash
cd /root/autodl-tmp/Code/SD_Guard/SD_5

DATASET_PATH=/root/autodl-tmp/SD_Guard/datasets/Qwen3GuardTest \
DRAFT_MODEL=/root/autodl-tmp/models/Huihui-Qwen3-0.6B-abliterated-v2 \
TARGET_MODEL=/root/autodl-tmp/models/Huihui-Qwen3-14B-abliterated-v2 \
GUARD_MODEL=/root/autodl-tmp/models/Qwen3Guard-Stream-4B \
GPU_ID=0 \
SAMPLE_COUNT=1 \
MAX_NEW_TOKENS=64 \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.72 \
OUTPUT_DIR=/root/autodl-tmp/sd4-smoke-cu130 \
bash scripts/run_matrix.sh \
  --local-files-only \
  --no-profile-vllm
```

只有在该命令验证了 Target、Draft、Guard、speculative hook、增量 Guard KV cache 和
标准 SD/inline Guard 输出等价性后，才能把这个环境标记为“SD_5 端到端验证通过”。

## 6. 正式实验

冒烟测试通过后再运行完整 2x4：

```bash
DATASET_PATH=/root/autodl-tmp/SD_Guard/datasets/Qwen3GuardTest \
DRAFT_MODEL=/root/autodl-tmp/models/Huihui-Qwen3-0.6B-abliterated-v2 \
TARGET_MODEL=/root/autodl-tmp/models/Huihui-Qwen3-14B-abliterated-v2 \
GUARD_MODEL=/root/autodl-tmp/models/Qwen3Guard-Stream-4B \
GPU_ID=0 \
OUTPUT_DIR=/root/autodl-tmp/sd4-results/sd4-cu130-2x4-100 \
bash scripts/run_matrix.sh --local-files-only
```

默认参数是 100 条、最大生成 4096 token、K=5、batch size 1、BF16、TP=1、
`gpu_memory_utilization=0.72`。

## 7. 跨服务器复用策略

### 同一台服务器或保留数据盘

直接复用 `/root/autodl-tmp/conda-envs/sd4-vllm-cu130`，不要每次运行重装环境。
保留 `UV_CACHE_DIR=/root/autodl-tmp/uv-cache` 后，重建环境时可复用已下载 wheel。

### AutoDL 保存镜像

当前已验证环境位于 `/root/autodl-tmp` 数据盘，AutoDL 保存系统镜像时通常不会把该
目录包含进去。若要依赖“保存镜像”复用 Python 环境，应在保存前把环境重新创建到
系统盘的 Conda envs 目录；模型和结果仍放数据盘。否则使用 AutoDL 的数据盘迁移功能。

### 其他 CUDA 13.0 服务器

复制代码仓库并运行第 3 节安装脚本。安装成功后记录：

```bash
python -m pip freeze > requirements-sd4-cu130.lock.txt
python -m pip check > pip-check.txt
CUDA_VISIBLE_DEVICES=0 python scripts/check_cuda130_env.py > sd4-env-summary.json
nvidia-smi > nvidia-smi.txt
```

锁文件用于记录“实际装了什么”，安装脚本用于描述“应该怎样装”。两者都要保留。
不同 GPU/驱动上的绝对速度不可直接归因于 SD4；Target-only、标准 SD 和 inline Guard
应在同一台服务器、同一环境内成组比较。
