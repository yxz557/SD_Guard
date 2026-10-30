# SD_Guard 项目主记忆

最后更新：2026-10-30

本文档记录项目当前已确认、后续协作必须遵守的仓库、路径、同步和 Python 安装约定。若实际环境发生变化，应先更新本文档，再修改运行脚本或实验说明。

## 1. GitHub 仓库

远程仓库：

```text
git@github.com:yxz557/SD_Guard.git
```

仓库为私有仓库，默认分支为 `main`。

Windows 本地项目路径：

```text
E:\Code Project\SD_Guard
```

Windows 本地路径没有变化，目录中新增了：

```text
.git/
.gitignore
```

Git 当前实际跟踪约 1969 个文件、36.84 MB。

## 2. GitHub 中不保存的内容

以下内容由 `.gitignore` 排除，不会上传 GitHub，也不会通过 `git clone` 下载到服务器：

```text
models/
results/
tmp/
任意层级的 outputs/
任意层级的 runs/
任意层级的 logs/
wandb/
.venv/
Python 缓存
模型权重
ZIP/TAR 压缩包
.env 和 API Key
本机专用配置
AgentDojo res/ 下的大量生成结果
```

模型文件扩展名也被全局排除：

```text
*.safetensors
*.pt
*.pth
*.ckpt
*.onnx
```

以下内容正常进入 GitHub：

```text
Code/ 源码
AgentDojo benchmark 数据
document/
thesis/
request/
普通 YAML 配置
普通 JSON/JSONL 数据
requirements.txt
pyproject.toml
uv.lock
```

注意：`tmp/` 下的临时 Python 脚本没有进入 GitHub。如果以后仍有用途，应移动到 `Code/scripts/` 后提交。

## 3. 服务器旧目录

原服务器项目目录：

```text
/home/model_and_data/yxz/Code/SD_Guard
```

计划改名保存为备份：

```text
/home/model_and_data/yxz/Code/SD_Guard_initial
```

执行方式：

```bash
cd /home/model_and_data/yxz/Code
mv -- SD_Guard SD_Guard_initial
```

暂时不要删除 `SD_Guard_initial`，至少等新 Git 仓库跑通 AgentDojo 后再处理。

## 4. 服务器新目录

从 GitHub 重新克隆：

```bash
cd /home/model_and_data/yxz/Code
git clone git@github.com:yxz557/SD_Guard.git SD_Guard
```

新项目根目录仍为：

```text
/home/model_and_data/yxz/Code/SD_Guard
```

但是其内部结构与原来手工上传的服务器目录不同。

旧 AgentDojo 路径：

```text
/home/model_and_data/yxz/Code/SD_Guard/FormalEXP/AgentLAB-main/Task-Injection/agentdojo
```

新的 AgentDojo 路径：

```text
/home/model_and_data/yxz/Code/SD_Guard/Code/FormalExp/AgentLAB-main/Task-Injection/agentdojo
```

变化有两点：

1. `SD_Guard` 后面增加了一层 `Code/`。
2. `FormalEXP` 改为 `FormalExp`。

Linux 区分大小写，必须使用 `FormalExp`，不能继续使用 `FormalEXP`。

可以查找确认：

```bash
find /home/model_and_data/yxz/Code/SD_Guard \
  -path "*/Task-Injection/agentdojo/pyproject.toml"
```

## 5. 日常同步流程

Windows 本地负责修改源码：

```powershell
cd "E:\Code Project\SD_Guard"
git status
git add -A
git commit -m "修改说明"
git push
```

L20 服务器负责拉取和运行：

```bash
cd /home/model_and_data/yxz/Code/SD_Guard
git pull --ff-only
```

推荐分工：

```text
Windows 本地：源码修改、commit、push
GitHub：代码中转和版本管理
L20 服务器：pull 代码、运行实验
```

尽量不要直接在服务器修改源码。如果确实需要修改，应提交并推送，避免 Windows 本地与服务器版本分叉。

## 6. 本地配置不会自动同步

以下文件被 `.gitignore` 排除：

```text
.env
.env.*
config.local.yaml
config.server.yaml
```

因此 API Key、服务器专用路径和本机配置需要分别配置。

例如每次新终端可能需要：

```bash
export LOCAL_LLM_PORT=8000
```

未来使用攻击生成 API 或 Guard API 时，还需要分别配置对应的 API Key。API Key 不得写入 Git 跟踪文件。

## 7. AgentDojo editable install

此前执行的命令：

```bash
python -m pip install -e .
```

其中 `-e` 表示 editable install，即可编辑安装。

### 7.1 普通安装和 editable 安装的区别

普通安装：

```bash
pip install .
```

可近似理解为把源码安装成环境中的固定副本：

```text
源码目录
   ↓ 复制/构建安装
site-packages/agentdojo
```

之后修改项目源码，环境中的包不一定同步更新，通常需要重新安装。

Editable 安装：

```bash
pip install -e .
```

它会让 Conda 环境记录 AgentDojo 源码的实际位置，可以近似理解为：

```text
sdg-agentdojo 环境
       ↓ 指向
/home/.../Task-Injection/agentdojo/src/
```

以后修改该目录中的 AgentDojo 源码，Python 会直接读取修改后的版本，通常不需要每次重新安装。

### 7.2 为什么目录变化后需要重新安装

此前 Conda 环境记住的可能是旧路径：

```text
/home/model_and_data/yxz/Code/SD_Guard/FormalEXP/AgentLAB-main/Task-Injection/agentdojo
```

旧目录改名且 GitHub 仓库重新克隆后，新路径变为：

```text
/home/model_and_data/yxz/Code/SD_Guard/Code/FormalExp/AgentLAB-main/Task-Injection/agentdojo
```

原 editable 安装记录的路径已经不正确，可能出现：

```text
ModuleNotFoundError: No module named agentdojo
```

也可能仍然意外加载 `SD_Guard_initial` 备份目录中的旧代码。

因此需要在新目录中重新执行一次：

```bash
conda activate sdg-agentdojo

cd /home/model_and_data/yxz/Code/SD_Guard/Code/FormalExp/AgentLAB-main/Task-Injection/agentdojo

python -m pip install -e .
```

这不是重新安装 CUDA、vLLM 或全部依赖，主要作用是把：

```text
agentdojo → 新源码路径
```

重新登记到当前 Conda 环境中。

验证 Python 实际加载位置：

```bash
python -c "import agentdojo; print(agentdojo.__file__)"
```

正确输出应包含：

```text
/home/model_and_data/yxz/Code/SD_Guard/Code/FormalExp/
```

不能再包含：

```text
SD_Guard_initial
```

### 7.3 为什么本项目推荐 editable 安装

后续仍可能修改：

- Qwen 本地接口；
- `LOCAL_LLM_PORT`；
- 模型枚举；
- 工具调用解析；
- 指标命名；
- Guard API 接入；
- 实验日志结构。

使用 editable 安装后：

```text
修改 AgentDojo 源码
→ 重新运行 benchmark
→ 修改立即生效
```

不需要每次执行完整的 `pip install`。

Editable install 不是复杂的“绑定操作”，只是告诉当前 Conda 环境：`agentdojo` 包应从新的源码目录直接加载。重新 clone 并确认新目录后执行一次 `python -m pip install -e .` 即可；后续普通的 `git pull` 修改源码通常会自动生效。

