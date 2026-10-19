# Trainning-Skill

面向 AI 编程助手的多机 GPU 训练检查 skill：在提交前发现可复现的配置错误，分配到资源后验证真实多机行为，并保留失败证据，减少反复退出、重新排队。

适用于新建多节点训练、更改节点数或 batch、恢复 checkpoint、排障分布式启动，以及确认 loss/吞吐是否正常。支持通用检查流程；附带的训练启动示例使用 **PyTorch torchrun**。

> 本仓库提供检查方法和示例，不包含模型训练实现或平台 SDK。检查通过不代表训练必然成功；平台 Running 也不代表已经完成参数更新。

## 主要内容

| 内容 | 作用 |
|---|---|
| [SKILL.md](multinode-training/SKILL.md) | AI 助手执行多机任务时的完整检查与验收流程 |
| [故障模式](multinode-training/references/failure-patterns.md) | DNS、网卡、IB、共享 checkpoint、W&B 和数据等待问题的排查与回归 |
| [check_plan.py](multinode-training/scripts/check_plan.py) | 纯数据并行的 batch、梯度累积、epoch 和保存里程碑计算 |
| [node_preflight.py](multinode-training/scripts/node_preflight.py) | 容器内 GPU、DNS、本地选路和 IB 用户态/设备打开检查 |
| [examples/](examples/) | 配置检查、每节点启动、后台监控、先提交再停止的 shell 示例 |
| [tests/](tests/) | 无需 GPU/网络的回归测试 |

## 安装与调用

```bash
git clone https://github.com/EricsXt/Trainning-Skill.git
cd Trainning-Skill
```

Codex 本机安装：

```bash
mkdir -p "$HOME/.codex/skills"
ln -s "$PWD/multinode-training" "$HOME/.codex/skills/multinode-training"
```

如果同名路径已经存在，先检查它的来源；不要直接覆盖其他版本。新开会话后，在请求中调用：

```text
使用 $multinode-training 检查本次2节点16卡训练。
每卡 microbatch=64，global batch=1024，从 base 开始训练20个 epoch。
先完成可执行检查；尚需实际 GPU 分配验证的项目明确标记出来。
不要停止现有排队任务。
```

其他支持 `SKILL.md` 的助手，可以把 `multinode-training/` 放入其技能目录；具体加载路径以该助手为准。

## 检查顺序与完成标准

1. **提交前检查**：实际入口、环境与依赖、挂载、数据和归一化、batch/恢复契约、共享目录并发测试。
2. **实际分配后检查**：当前节点 DNS/网卡、可打开的 IB 设备、设备数量、真实跨节点 collective。
3. **训练验收**：所有 rank 完成实际更新，loss 等指标有限且持续推进，目标 batch 不 OOM。
4. **保存与恢复验收**：生产 checkpoint 路径保存、完整性读取、恢复后更新通过。
5. **持续观察**：区分编译/预热和稳定耗时，记录 samples/s、数据等待、显存和退出码。

单进程检查不能证明多机共享目录安全；设备打开不能证明 NCCL 可通信；collective 成功不能证明训练/保存正确。上述阶段分别记录证据，不合并成一句“全部检查通过”。

## 快速验证 batch 计划

从仓库根目录执行，仅需要 Python 3 标准库：

```bash
python3 multinode-training/scripts/check_plan.py \
  multinode-training/references/plan-example.json
```

示例为2节点×8卡，每卡64、梯度累积1，全局 batch 1024。以1,378,118个采样锚点为例：

- 每个 epoch 1345次 optimizer update，drop_last 丢弃838个样本。
- 20个 epoch 共26,900次更新；每个 epoch 保留 checkpoint。
- warmup 150步，对应153,600个样本的暴露量。

这是算术示例，不是通用模型的推荐超参数。真实 dataloader 的尾批、累积和重采样语义必须另验。检查器仅支持纯数据并行，混合并行或不丢弃尾批时会拒绝套用这个公式。

## Shell 示例

示例默认不连接任何训练平台，不包含账号、Token、内部域名或真实集群 ID。除 `replace_job.sh` 的已配置 adapter 外，没有自动提交、停止、重试或删除 checkpoint 的操作。

依赖：Linux、Bash、Python 3；物理 GPU 检查需要 `nvidia-smi`，IB 检查需要对应用户态库/provider及设备权限，持续监控锁需要 `flock`。训练框架和数据依赖由你的环境提供。

### 1. 配置检查：preflight.sh

复制并编辑 [training.env.example](examples/training.env.example)，填入所有节点都能访问的绝对路径：

```bash
mkdir -p "$HOME/.config/training"
cp examples/training.env.example "$HOME/.config/training/training.env"
chmod 600 "$HOME/.config/training/training.env"
# 编辑上述文件，替换 /shared 下的示例路径。
export TRAINING_ENV_FILE="$HOME/.config/training/training.env"
bash examples/preflight.sh --config
```

训练计划 JSON 可从 [plan-example.json](multinode-training/references/plan-example.json) 复制修改。此模式检查路径和计划，不启动训练。它不会替代实际数据读取或 checkpoint 并发测试。

实际分配到节点后，按该次平台环境设置 `MASTER_ADDR`、`NODE_RANK`，在每个节点运行：

```bash
bash examples/preflight.sh --node
```

将私有配置放在工作节点可访问的位置，或直接使用平台环境变量。开发机的 `$HOME` 文件不一定在训练容器内存在。示例只 source 明确指定的可信配置，不读取整份 `.zshrc`。

### 2. 每节点启动：torchrun_multinode.sh

平台以**每个节点一次**的方式执行以下命令。平台提供当前主节点地址、节点 rank；配置中指定 `NNODES=2`、`GPUS_PER_NODE=8`：

```bash
TRAINING_ENV_FILE=/shared/private/training.env \
  bash /shared/Trainning-Skill/examples/torchrun_multinode.sh \
  --config /shared/project/train.yaml
```

`--config` 及后续参数原样传给你自己的 `TRAIN_SCRIPT`；它们是否存在取决于你的训练程序。脚本会先执行节点 preflight，再 `exec torch.distributed.run`，保留训练退出码和信号。

**batch、学习率、epoch、保存策略由你的训练程序/配置设置**，不会因 shell 中有16张卡就自动正确。`PLAN_PATH` 应与实际训练配置对应，程序内也应记录实际 batch。

本示例会拒绝嵌套 `LOCAL_RANK` 启动，不硬编码 bond0/HCA，不自动把 IB 改成 socket，默认 `NCCL_DEBUG=WARN`。需要诊断时可临时开启 INFO，但应控制日志体积。

JAX/OpenPI 的“一节点一个 controller、管理本节点多张卡”入口与 torchrun 不同，**不要直接用此脚本包住 JAX 训练**。使用对应项目的 JAX 入口，按 skill 验证其实际 process/device 数、collective和共享 checkpoint 路径。

### 3. 后台只读观察：watch_logs.sh

单次查看两个节点日志的最近片段：

```bash
bash examples/watch_logs.sh --once /shared/run/train.node0.log /shared/run/train.node1.log
```

后台每30秒采样，断开终端后继续：

```bash
mkdir -p /shared/private/monitor
MONITOR_LOCK=/shared/private/monitor/run.lock INTERVAL_SECONDS=30 \
  nohup bash examples/watch_logs.sh --follow \
  /shared/run/train.node0.log /shared/run/train.node1.log \
  > /shared/private/monitor/run.log 2>&1 < /dev/null &
echo "$!" > /shared/private/monitor/run.pid
```

每次每个文件最多读取256 KiB，使用同一 run 的锁防止重复启动；不控制训练进程，也不自动推送消息。大量 NCCL 输出可能挤掉尾部的指标，因此它只辅助诊断，完整指标应由训练端增量解析或 W&B history 获取。停止监控前先核对保存的 PID，避免 PID 复用后误杀其他进程。

### 4. 先提交再停旧任务：replace_job.sh

这是**会产生平台操作**的模板，只有在替换动作已获授权且 adapter 配置完成后执行：

```bash
PLATFORM_ADAPTER=/shared/private/platform_adapter \
REQUEST_FILE=/shared/private/validated-request.json \
OLD_JOB_ID=the-exact-old-job-id \
  bash examples/replace_job.sh
```

不同平台 API 不同，本仓库不猜测其接口。你需要实现可执行的 `PLATFORM_ADAPTER`，满足以下标准化协议；JSON走 stdout，诊断走 stderr，失败返回非零退出码：

| 调用 | 标准化输出 |
|---|---|
| `submit REQUEST_FILE` | `{"job_id":"new-id"}` |
| `detail JOB_ID` | `{"job_id":"new-id","status":"Inqueue","configuration_verified":true}` |
| `stop JOB_ID` | `{"job_id":"old-id","stopped":true}` |

`configuration_verified=true` 必须表示已对照提交请求核实资源、命令、挂载、优先级等实际读回值，不能直接写死。stop adapter 需轮询确认目标已停止后才返回 `stopped=true`。

脚本顺序为提交 → 读回并验证新任务 → 停止指定旧任务。提交/读回失败时保留旧任务；新 ID 与旧 ID 相同会拒绝停止。如果提交已成功但之后网络失败，先检查平台是否已有新任务，**不要盲目重跑产生重复队列**。模板不提供事务性回滚。

## 验证本仓库

```bash
python3 -m unittest discover -s tests -v
for script in examples/*.sh; do bash -n "$script"; done
```

回归覆盖 batch/epoch 算术、DNS恢复/超时，以及 shell 示例的参数传递、锁、替换顺序和失败保护。测试使用临时目录与 mock adapter，不会申请 GPU、提交平台任务或停止真实训练。

这些测试不证明 IB 性能、真实模型显存峰值或 checkpoint恢复正确；这些需要在已授权的实际资源上验证。

## 凭据与版本记录

- 密钥只放在平台 secret 或私有文件，不写进示例、命令历史或提交记录。
- 本地诊断可能含主机名、IP、设备和路径，报告留在私有工作区，不默认上传。
- 记录源码/SDK版本、计划、数据及环境依赖；每次尝试使用独立输出和 W&B run。
- 对失败任务先查首个异常，再完成对应回归；不要把后续通信超时当作唯一根因，也不要靠反复重提试运气。
