# 多机启动失败：证据、修复与回归

| 现象 | 应检查的首个证据 | 修复与回归 |
|---|---|---|
| `Name or service not known` | 具体 hostname、注入的 MASTER/PET_MASTER 值、Pod/服务/DNS 是否就绪 | 对 EAI_AGAIN/EAI_NONAME 有限重试；测试短暂错误恢复、永久错误超时和正常地址；仍失败调查域名而非无限重提。该错误本身不能证明只是时序问题。 |
| `no socket interface found` | 容器内实际接口、地址和路由；显式 NCCL_SOCKET_IFNAME | 从本次分配选可用接口；测试不同节点接口名称以及无效显式覆盖；不要照抄文档中的 bond0。 |
| `Failed to open libibverbs.so` / `Failed to initialize NET plugin IB` | 用户态库、provider、依赖 ABI、容器 /dev 暴露、实际设备打开 | 修复镜像/挂载运行时，检查 transitive dependencies；双节点真实 collective 验收。安装了 Python 包不代表具备 IB 系统库。 |
| 文档中的 HCA 名称均不可用 | sysfs 设备是否真的映射到容器；active port；ibv_open_device 结果 | 仅选已分配、可打开、active 的设备；显式配置错误不悄悄替换。 |
| 首次 base 启动报 checkpoint `FileExistsError`，resume 却成功 | 所有 rank 是否独立查询共享目录后决定 resume/overwrite | 一致的恢复决策或协调创建；真实多进程覆盖新目录、空目录、有效断点，不能仅做单进程 syntax check。 |
| 一节点已退出，另一节点仍 Running | 首个 traceback、训练 PID、各 rank exit code、collective 超时 | 优先处理原始异常；后续 shutdown timeout/exit 134 往往是次生错误。不要只修退出码，也不要称另一节点仍在正常训练。 |
| W&B 有 GPU 指标但没有 loss | 原始训练日志是否已有标量，日志是否缓冲，logger 的文件偏移和 run ID | 未输出标量时定位训练阶段；已有标量则检查 flush/解析/上传。正确处理从断点恢复后的横轴，不能把恢复步数当新进度。 |
| batch 增大后 step 变慢 | actual global batch、真实 optimizer update、稳定窗口样本吞吐、数据等待 | 用 samples/s、每 epoch 总耗时和训练效果比较；改变 batch 后重算调度，不宣称显存利用率等于速度。 |
| 恢复后数据重复/跳跃 | 样本游标是否以旧 batch 编码；随机种子和 sampler epoch | 验证每 rank 样本分片及游标；做显式迁移或独立从 base 开始，不能直接重用 step。 |
| 起初正常，随后慢或存储受压 | NCCL 日志增长率、监控读取量、视频解码、动作 JSON fallback、尾延迟 | 只读监控用增量/有界读取；按需减少 verbose 日志；先测瓶颈，不在运行中盲目替换数据格式。 |

## 因果判断规则

- “修改后通过”不是原始原因的唯一证明。保留失败日志、配置差异和对应回归范围。
- 本机 CPU 多进程测试能发现共享目录竞态，不能证明分配后的 IB 或 GPU 容量。
- 相同镜像/脚本在不同分配可遇到不同网卡、DNS 时序和设备可见性；每次物理分配仍需轻量 preflight。
- 每次新节点、精度、batch、初始化方式或并行模式改变，明确哪些旧证据仍适用，哪些需重新验证。
