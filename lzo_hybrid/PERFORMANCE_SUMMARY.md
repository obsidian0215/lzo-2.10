# LZO Hybrid 性能总结（Intel + Nvidia）

<!-- markdownlint-disable-file MD012 -->

> 代码路径：`/root/lzo-2.10/lzo_hybrid`
> 当前二进制：`/root/lzo-2.10/lzo_hybrid/lzo_hybrid`
> 当前哈希：`sha256=1fa2831f6fab1f3a93a4252ba7bee8684600bf9b425691f303aedaa32932dbe7`
> 基线二进制：`/root/lzo-2.10/exp_results/baselines/lzo_hybrid_baseline_round22f`
> 基线哈希：`sha256=d58b3440218f4e2956ef922362c73e0d3ff807dd0ee28a14cbf20b9483adf665`

---

## Intel 平台（五章结构）

### 1. 设计动机

#### 1.1 为什么不是“单引擎继续打磨”

`lzo_hybrid` 的存在目的，不是简单把 `lzo_gpu` 和 `lzo_cpu` 拼在同一可执行文件里，而是建立一个“在同一容器语义下可重复评估 CPU/GPU 协作收益”的执行体系。

单引擎优化的问题在于：

1. 很难在同一输入集上区分“计算增益”与“主机侧协同成本”；
2. 很难解释“为什么某轮是压缩正向、解压负向”；
3. 很难把调度策略、块划分策略、设备特征放到同一模型里统一解释；
4. 很难做 fixed/adaptive 两条路线的可重复对照。

Hybrid 设计正是为了解决这四个问题。

#### 1.2 本文的目标不是“报结果”，而是“可解释”

本文明确把重点放在三个层面：

1. **系统设计可解释**：为什么这样分层、为什么是 prefix-only、为什么 fixed/adaptive 共用同一容器解释；
2. **执行过程可解释**：从输入读取到容器输出，从容器解析到解压回放，每一步如何落到代码符号；
3. **优化实现可解释**：每项优化在哪里、解决什么问题、如何验证它没有破坏语义。

你要的是“看得懂”，所以本文不会只给短句结论，而是把关键逻辑展开到可直接对照源码的粒度。

#### 1.3 当前实现边界（这部分必须先讲清）

当前主线实现已经收敛为：

- 仅保留 prefix 分区语义；
- `partition_blocks()` 只转发到 `partition_blocks_prefix()`；
- 容器解释不再依赖额外分区模式字段；
- 压缩与解压共享同一边界语义。

这意味着本文所有“分路”描述都只讨论 prefix 边界，不再混入其它布局口径。

#### 1.4 评估口径与约束

本线评估采用如下约束：

1. 主指标为 `CompTotal / DecTotal`；
2. `Ratio` 变化必须并列观察；
3. roundtrip 正确性是硬门禁；
4. 结论必须绑定到函数与工件路径。

这也是你反复强调“不要写空话”的核心要求。

---

### 2. 系统架构

#### 2.1 分层与职责

`lzo_hybrid` 采用“入口层—核心层—结构层—内核层”四层组织：

| 层级 | 文件 | 职责 |
| --- | --- | --- |
| 入口层 | `lzo_hybrid.c` | 参数解析、模式分流、OpenCL 初始化、bench 驱动 |
| 核心层 | `lzo_hybrid_core.c` | 分块、分区、并行编排、容器读写、阶段统计 |
| 结构层 | `lzo_hybrid_core.h` | 参数对象、工作区对象、阶段统计对象定义 |
| 内核层 | `../lzo_gpu/*.cl` | GPU压缩、GPU解压、pack kernel |

这种分层的目的，是把“策略决策”与“执行细节”解耦：

- `lzo_hybrid.c` 决定 **要不要走 Hybrid、走哪种模式**；
- `lzo_hybrid_core.c` 决定 **具体怎么切块、怎么并行、怎么回收结果**。

#### 2.2 执行对象与数据对象

核心执行对象：

1. `hybrid_params_t`：本轮策略与配置；
2. `hybrid_workspace_t`：GPU缓冲与容量复用状态；
3. `hybrid_timing_t`：分阶段统计。

核心数据对象：

1. 输入原始缓冲；
2. 块长度表 `comp_lengths[]`；
3. 偏移表 `offsets[]`；
4. 压缩payload；
5. 解压输出缓冲。

#### 2.3 容器语义（必须精讲）

容器头部字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `magic` | `uint16_t` | 容器魔数 |
| `orig_sz` | `uint32_t` | 原始输入大小 |
| `blk_sz` | `uint32_t` | 块大小 |
| `nblk` | `uint32_t` | 块数量 |
| `alg_id` | `uint32_t` | 算法编号 (`lzo1x` / `lzo1y`) |
| `comp_lengths[]` | `uint32_t[]` | 每块压缩长度 |

容器尾部是按块拼接的压缩payload。解压时先恢复长度表，再由长度表构造偏移表，最后按块回放。

这里最关键的是：**容器语义与分区策略是正交关系**。换句话说，CPU/GPU 谁压了哪个块，不改变容器解释本身。

#### 2.4 prefix-only 分区如何落地

prefix-only 的边界定义很直接：

- GPU处理 `0..gpu_count-1`；
- CPU处理 `gpu_count..nblk-1`。

在代码里，这通过 `partition_blocks_prefix()` 生成块索引数组（或前缀快路径下直接按连续区间处理）来落地。

这一点非常重要：解压端沿用同一边界解释，避免了“压缩按A边界写、解压按B边界读”的语义漂移。

#### 2.5 调用链总览

压缩入口链：

- `main()` -> `hybrid_compress()` -> `hybrid_compress_buf()`

解压入口链：

- `main()` -> `hybrid_decompress()` -> `hybrid_decompress_buf()`

bench入口链：

- `main()` -> `hybrid_bench()`

自适应比例链：

- `choose_adaptive_gpu_ratio()`
- `hybrid_adaptive_adjust_gpu_blocks()`

---

### 3. 核心设计和优化

#### 3.1 压缩过程（端到端展开）

下面按真实执行顺序拆开压缩路径。

##### 3.1.1 输入与分块阶段

1. 读取输入文件到内存；
2. 调用 `hybrid_choose_blocking()` 计算 `blk/nblk`；
3. 对 auto block 模式施加最小块规则；
4. 计算 `worst_blk = lzo_worst_size(blk)`。

这一阶段决定了后续所有数组尺寸与 kernel 参数尺寸。

##### 3.1.2 比例决策阶段

比例来源分两类：

- 固定比例：直接使用 `params->gpu_ratio`；
- 自适应比例：`choose_adaptive_gpu_ratio()` 给出，再由 `hybrid_adaptive_adjust_gpu_blocks()` 做块数层面修正。

修正的核心目标是避免在小块数场景产生抖动边界。

##### 3.1.3 分区阶段

分区入口 `partition_blocks()` 只调用 `partition_blocks_prefix()`，最终得到：

- `gpu_idx/gpu_count`
- `cpu_idx/cpu_count`

在前缀快路径下，索引数组可以省略，直接按连续区间执行。

##### 3.1.4 CPU路径阶段

CPU侧由 `cpu_compress_worker()` 执行实际压缩，线程池由 `cpu_compress_pool_t` 驱动。

关键点：

1. 任务领取使用原子 `next_idx`；
2. 每线程独立 `wrkmem` 切片；
3. 每块压缩长度回填 `lengths[]`；
4. 失败通过 `rc=-1` 汇总上抛。

##### 3.1.5 GPU路径阶段

GPU侧关键步骤：

1. grow-only 复用 `d_in/d_out/d_len`；
2. 分配并管理 `d_dict`；
3. 非999路径下维护 `comp_epoch_base`；
4. 设置 kernel 参数并发射 NDRange；
5. 回读长度表。

这里的重点优化不是“单次跑得快”，而是“多轮 bench 不反复分配导致抖动”。

##### 3.1.6 pack/非pack 分支

`hybrid_should_use_device_compaction()` 决定是否走 pack：

- pack 开：构造 packed offsets，发射 pack kernel，回读 packed payload；
- pack 关：直接从 `d_out` 回读按块槽位数据。

这一步解决的是“稀疏槽位写回字节过大”的问题，但只在收益达标时启用。

##### 3.1.7 输出组装阶段

最终由 `lzo_write_compressed_file()` 写容器：

1. 写头字段；
2. 写长度表；
3. 写payload。

容器层不关心块来自 CPU 还是 GPU。

#### 3.2 解压过程（端到端展开）

##### 3.2.1 解析阶段

1. 校验魔数；
2. 读取 `orig_sz/blk_sz/nblk/alg_id`；
3. 读 `comp_lengths[]`；
4. 构建 `offsets[]`；
5. 读压缩payload。

##### 3.2.2 比例与分区阶段

解压同样支持 fixed/adaptive，且边界修正规则一致。

这保证了“同一参数配置下压缩与解压具有相同分区语义”，减少理解成本。

##### 3.2.3 CPU解压阶段

`cpu_decompress_worker()` 根据块索引回放：

1. 计算每块输出偏移；
2. 调用 `lzo1x_decompress_safe` 或 `lzo1y_decompress_safe`；
3. 验证输出长度与期望块长度一致；
4. 异常统一上抛。

##### 3.2.4 GPU解压阶段

GPU解压关键步骤：

1. 上传 `comp/off/comp_lens`；
2. 分配 `d_decomp_out` 与 `d_out_lens`；
3. 发射 `*_block_decompress` kernel；
4. 回读输出并按分区语义写回目标缓冲。

prefix路径下可直接连续回写，非prefix映射路径按块索引回放。

##### 3.2.5 写出阶段

把 `output_buf` 写入目标文件，或在 `/dev/null` 模式跳过写盘。

#### 3.3 自适应模型（实现级解释）

`choose_adaptive_gpu_ratio()` 的核心思路不是拍脑袋，而是把分配比例建模为“设备能力 + 数据特征 + 运行态”三项乘积。

设备能力：

- `Pc0`（CPU基线吞吐）
- `Pg0`（GPU基线吞吐）
- `t0`（GPU固定开销）

数据特征：

- 熵采样得到 `gC/gG` 校正因子

运行态：

- `sC/sG`（CPU/GPU可用度）

最后在比分上叠加能效权重和解压主机协调修正。

#### 3.4 工作区复用策略

`hybrid_workspace_t` 的核心价值是把“每轮临时分配”改成“容量扩展 + 长驻复用”。

策略要点：

1. 只增不减；
2. 扩容后更新容量元数据；
3. bench 多轮直接复用。

这能显著降低驱动层反复建对象引发的波动。

#### 3.5 阶段统计对象的意义

`hybrid_timing_t` 不是装饰字段，而是定位问题的主工具。

关键字段解释：

- `gpu_kernel_us`：GPU核函数窗口；
- `cpu_kernel_us`：CPU工作线程窗口；
- `upload_us/download_us`：主机—设备搬运窗口；
- `total_us`：整体阶段窗口。

当“kernel正向但total不明显”时，这组字段能直接定位问题落点。

#### 3.6 失败路径设计

失败路径统一目标：

1. 不写脏输出；
2. 不泄露资源；
3. 不吞错误码。

对应策略：

- 所有中间缓冲在失败分支释放；
- 线程对象 join 后清理；
- `rc` 统一汇总上抛。

#### 3.7 adaptive 分配机制详解

这一节单独展开你点名的 adaptive 机制，回答四个问题：

1. adaptive 的目标到底是什么；
2. adaptive 是如何计算比例的；
3. adaptive 是如何从连续比例映射成离散块数的；
4. adaptive 在边界场景如何避免抖动。

##### 3.7.1 adaptive 的目标

adaptive 的目标不是“让 GPU 比例尽可能大”，而是“在当前输入特征与设备状态下，让整体阶段值更稳，且不破坏容器语义”。

更具体地说，它追求的是：

- 让 CPU 与 GPU 的并行窗口尽量接近平衡；
- 避免在小输入上因为固定开销过大导致纯负收益；
- 在多轮运行中减少比例震荡。

##### 3.7.2 adaptive 的输入变量

`choose_adaptive_gpu_ratio()` 会使用以下变量：

1. 设备画像变量：`Pc0`, `Pg0`, `t0`；
2. 数据校正变量：`gC`, `gG`；
3. 运行态变量：`sC`, `sG`；
4. 线程变量：`thread_count`；
5. 数据规模变量：`B`。

其中：

- `Pc0`：CPU 基线吞吐；
- `Pg0`：GPU 基线吞吐；
- `t0`：GPU 固定开销；
- `gC/gG`：由采样特征推导的修正系数；
- `sC/sG`：当前可用度因子。

##### 3.7.3 adaptive 的计算骨架

计算流程分三层：

第一层：得到有效吞吐

- `Pc_eff = Pc0 * gC * sC * thread_count`
- `Pg_eff = Pg0 * gG * sG`

第二层：得到基础比例

- `r_base = Pg_eff / (Pc_eff + Pg_eff)`

第三层：做开销修正与约束

- 减去 `t0` 对小规模输入的影响；
- 限制到 `[0,1]`；
- 进入块数级修正。

##### 3.7.4 为什么要有块数级修正

比例是连续值，但执行单位是离散块。若直接四舍五入，可能出现两个问题：

1. 混合分配时某一侧块数太少，导致协同开销大于收益；
2. 相邻输入在边界处反复左右跳，产生抖动。

`hybrid_adaptive_adjust_gpu_blocks()` 通过最小混合块门限与量化步长进行修正，目的就是把“比例决策”变成“可执行且稳定的块级决策”。

##### 3.7.5 adaptive 与 prefix-only 的关系

adaptive 只决定“GPU 前缀长度”，不改变分区语义本身。

因此：

- fixed 与 adaptive 都落到 prefix 解释；
- 容器层不感知 adaptive/fixed；
- 解压不需要额外分区字段。

##### 3.7.6 adaptive 的退化策略

在以下场景会退化到单侧执行：

1. 输入规模过小；
2. `cpu_threads <= 0`；
3. GPU 路径不可用；
4. 块数修正后混合收益不足。

这种退化是“显式策略”，不是“异常行为”。

#### 3.8 pack 机制（逐层展开）

这一节单独展开你点名的 pack 机制，回答四个问题：

1. pack 在 Hybrid 里解决什么问题；
2. pack 何时开启；
3. pack 开启后的数据路径是什么；
4. pack 关闭时回退路径是否语义等价。

##### 3.8.1 pack 的目标

GPU 压缩输出在 `d_out` 中按“固定槽位”布局，槽位大小是 `worst_blk`。这种布局便于并行写入，但在很多输入上会产生稀疏浪费。

pack 的目标是：

- 把稀疏槽位压紧成连续payload；
- 降低回读字节量；
- 降低主机侧拷贝与组装成本。

##### 3.8.2 pack 开启判定

`hybrid_should_use_device_compaction()` 不是单条件判定，而是联合判定：

1. `pack_kernel` 可用；
2. `packed_bytes < sparse_bytes`；
3. `num_blocks` 达到最小门限；
4. 节省比例达到门限。

只有全部满足才开启。

##### 3.8.3 pack 开启后的数据路径

路径分五步：

1. 从 `lengths[]` 构建 packed offsets；
2. 上传 offsets 到 `d_packed_off`；
3. 发射 pack kernel，把 `d_out` 变成 `d_packed_out`；
4. 回读连续 packed payload；
5. 写入容器 payload。

##### 3.8.4 pack 关闭路径

pack 关闭时，回读逻辑仍保证语义一致：

- prefix 连续路径可直接读回；
- 非prefix映射路径按块索引回放；
- 最终 payload 与 `lengths[]` 一致。

##### 3.8.5 pack 与容器解释关系

pack 只改变“内部搬运形态”，不改变“容器解释形态”。

容器仍然是：

1. 头；
2. 长度表；
3. payload。

因此 pack 开关不会改变解压语义。

#### 3.9 压缩执行状态机（实现级）

为了把过程讲透，这里给出压缩状态机：

1. `S0` 读取输入；
2. `S1` 计算分块；
3. `S2` 比例决策；
4. `S3` 前缀分区；
5. `S4` CPU线程池启动；
6. `S5` GPU缓冲准备；
7. `S6` GPU kernel 发射；
8. `S7` 回读长度；
9. `S8` pack判定；
10. `S9` pack/非pack回读；
11. `S10` 等待CPU收敛；
12. `S11` 容器组装；
13. `S12` 输出写盘；
14. `S13` 资源回收。

任何阶段失败都进入失败回滚分支，并保证不输出脏文件。

#### 3.10 解压执行状态机（实现级）

解压状态机如下：

1. `D0` 打开输入并读头；
2. `D1` 校验头字段；
3. `D2` 读取长度表；
4. `D3` 构建偏移表；
5. `D4` 比例决策；
6. `D5` 前缀分区；
7. `D6` CPU线程池启动；
8. `D7` GPU解压参数绑定；
9. `D8` GPU kernel 发射；
10. `D9` 回读并回放到输出缓冲；
11. `D10` 等待CPU收敛；
12. `D11` 输出写盘；
13. `D12` 资源回收。

关键点是 `D4~D10`：压缩与解压使用同一分区语义，避免行为分叉。

#### 3.11 代码映射（函数级）

这一节只做“概念 -> 代码符号”映射，方便你直接审代码。

##### 3.11.1 比例与分区映射

| 设计概念 | 代码符号 |
| --- | --- |
| fixed 比例入口 | `params->gpu_ratio` |
| adaptive 比例入口 | `choose_adaptive_gpu_ratio` |
| 块数修正 | `hybrid_adaptive_adjust_gpu_blocks` |
| 分区入口 | `partition_blocks` |
| prefix分区实现 | `partition_blocks_prefix` |

##### 3.11.2 CPU压缩映射

| 设计概念 | 代码符号 |
| --- | --- |
| 压缩 worker | `cpu_compress_worker` |
| 线程池对象 | `cpu_compress_pool_t` |
| 任务对象 | `cpu_compress_job_t` |
| 算法调用(1x) | `lzo1x_1_compress` |
| 算法调用(1y) | `lzo1y_1_compress` |

##### 3.11.3 GPU压缩映射

| 设计概念 | 代码符号 |
| --- | --- |
| 输入缓冲复用 | `grow_buffer(..., d_in, ...)` |
| 输出槽位缓冲 | `grow_buffer(..., d_out, ...)` |
| 长度缓冲 | `grow_buffer(..., d_len, ...)` |
| 字典缓冲 | `grow_buffer(..., d_dict, ...)` |
| compaction判定 | `hybrid_should_use_device_compaction` |
| pack kernel | `lzo_pack_compressed_blocks` |

##### 3.11.4 CPU解压映射

| 设计概念 | 代码符号 |
| --- | --- |
| 解压 worker | `cpu_decompress_worker` |
| 线程池对象 | `cpu_decompress_pool_t` |
| 任务对象 | `cpu_decompress_job_t` |
| 算法调用(1x) | `lzo1x_decompress_safe` |
| 算法调用(1y) | `lzo1y_decompress_safe` |

##### 3.11.5 GPU解压映射

| 设计概念 | 代码符号 |
| --- | --- |
| 压缩输入缓冲 | `d_comp` |
| 偏移缓冲 | `d_off` |
| 压缩长度缓冲 | `d_comp_lens` |
| 解压输出缓冲 | `d_decomp_out` |
| 输出长度缓冲 | `d_out_lens` |
| 解压 kernel | `*_block_decompress` |

##### 3.11.6 阶段统计映射

| 设计概念 | 代码符号 |
| --- | --- |
| CPU核窗口 | `cpu_kernel_us` |
| GPU核窗口 | `gpu_kernel_us` |
| 上传窗口 | `upload_us` |
| 下载窗口 | `download_us` |
| 整体窗口 | `total_us` |

#### 3.12 你关心的“每一个点到底怎么做”的补充说明

##### 3.12.1 adaptive 到底做了什么

它做了三件事：

1. 用设备画像与运行态估算双侧有效能力；
2. 计算一个连续比例；
3. 把连续比例稳定映射成离散块边界。

##### 3.12.2 pack 到底做了什么

它做了四件事：

1. 用长度表计算紧凑偏移；
2. 设备端打包稀疏槽位；
3. 把打包结果连续回读；
4. 写入同一容器语义。

##### 3.12.3 prefix-only 到底约束了什么

它约束的是“块归属边界只能是一刀切前缀”，而不是“只能全GPU或全CPU”。

因此：

- 既能混合分配；
- 又能保持容器解释简洁；
- 还能让解压端不需要额外分区元数据。

##### 3.12.4 为什么你会觉得之前版本看不懂

之前版本的问题确实在于：

1. 对 adaptive/pack 只写了短结论，没有把输入、输出和边界讲完整；
2. 对执行过程只写了主路径，没有写状态机与异常回滚；
3. 对代码映射不够细，读者难以从文档跳到函数。

本次改写就是针对这三点做补齐。

#### 3.13 adaptive 机制（实现级全展开）

这一节不再给“高层解释”，而是严格按 `lzo_hybrid_core.c` 的执行顺序，把 adaptive 的每一步拆到变量级。

##### 3.13.1 入口与短路条件

adaptive 主入口是：

- `choose_adaptive_gpu_ratio(...)`
- 随后进入 `hybrid_adaptive_adjust_gpu_blocks(...)`

在入口阶段有三个必须先说清的短路：

1. `gpu_kernel == NULL`：直接返回 `0.0`（CPU-only）；
2. 参数不完整（`params==NULL` 或 `nblk==0` 或 `blk==0` 或 `total_input_sz==0`）：返回 `0.5`；
3. 后续若开销判定触发（`B <= t0 * Pg_eff`）：再次强制 `0.0`。

这三条是 adaptive 的“硬护栏”，保证不会把无效输入推进复杂模型。

##### 3.13.2 第一步：设备画像校准（`lzo_calibrate_device_profile`）

adaptive 不是每次都做重校准，而是走全局缓存 `g_lzo_dev_profile`：

1. CPU 画像：
	- 2MB 校准缓冲；
	- 同算法路径（`lzo1x_1_compress` 或 `lzo1y_1_compress`）执行 3 次；
	- 得到 `cpu_throughput`（字节/秒）；
	- 若可读 RAPL，估算 `cpu_energy_per_byte`。
2. GPU 画像：
	- 同样 2MB 输入走 GPU kernel；
	- 记录 `gpu_throughput`；
	- `gpu_overhead_s` 当前设定为 `0.001` 秒；
	- 若可读 RAPL 域，估算 `gpu_energy_per_byte`。
3. 回退默认值：
	- CPU 吞吐默认 `300e6`；
	- GPU 吞吐默认 `1500e6`；
	- 开销默认 `0.001`。

这一步完成后，adaptive 才有 `Pc0/Pg0/t0` 的基础。

##### 3.13.3 第二步：数据特征采样（熵 + CPU样本吞吐）

采样不是“连续头部采样”，而是通过 `sampled_block_index()` 在块空间均匀抽样：

$$
	ext{blk\_idx} = \frac{\text{sample\_pos} \cdot (nblk-1)}{(sample\_count-1)}
$$

实现里还有两个细节：

1. `prev_block` 去重，避免边界导致重复索引；
2. 对每个样本块同时做两件事：
	- `lzo_calc_entropy(...)` 统计熵；
	- 实际调用一次 CPU 压缩，累计样本吞吐。

因此 adaptive 的数据特征不是“只看熵”，而是“熵 + 真实样本压缩时间”双源输入。

##### 3.13.4 `gC` 与 `gG` 的具体计算

先看 `gC`（CPU数据因子）：

$$
g_C = \text{clamp}\left(\frac{\text{sample\_cpu\_throughput}}{P_{c0}},\ 0.3,\ 3.0\right)
$$

再看 `gG`（GPU数据因子），代码里的映射是：

1. 熵归一：$R_{est}=\text{entropy}/8$，再夹紧到 `[0.05, 0.95]`；
2. 使用常量 `m=2.0`, `Rref=0.50`：

$$
g_G = \frac{1+m+R_{ref}}{1+m+R_{est}},\quad g_G\in[0.5,2.0]
$$

这个设计的意义是：

- 熵变化对 GPU 比例有影响；
- 影响被限定在可控区间，不会因为极端样本把比例拉爆。

##### 3.13.5 运行态因子 `sC/sG` 与线程修正

`sC` 来自 `/proc/stat` 的可用度估算，`sG` 来自 `gpu_busy_percent`。两者都最终压到 `[0.05,1.0]` 或 `[0,1]` 的合理区间。

此外有一条“线程修正”：

- 当 `params->cpu_threads < total_cores` 时，`sC` 乘以 `total_cores/cpu_threads` 再上限到 `1.0`；
- 目的是避免“人为限制线程数”被误解成“CPU本身很忙”。

##### 3.13.6 主公式：连续比例 `r_star`

有效吞吐：

$$
P_{c,eff}=P_{c0}\cdot g_C\cdot s_C\cdot \text{thread\_count}
$$

$$
P_{g,eff}=P_{g0}\cdot g_G\cdot s_G
$$

基础比例：

$$
r_{base}=\frac{P_{g,eff}}{P_{c,eff}+P_{g,eff}}
$$

开销修正后：

$$
r^*=r_{base}-\frac{t_0\cdot P_{c,eff}\cdot P_{g,eff}}{B\cdot(P_{c,eff}+P_{g,eff})}
$$

其中 $B=\text{total\_input\_sz}$。这正是代码注释中的 makespan 思路。

##### 3.13.7 小输入退化判定

若满足：

$$
B \le t_0\cdot P_{g,eff}
$$

则直接返回 CPU-only（`0.0`）。

这条规则非常关键：它不是“GPU慢”，而是“固定开销在该输入规模下无法摊薄”。

##### 3.13.8 能效纠偏（70/30）

代码里性能与能效的缺省权重是：

- `perf_weight_pct = 70.0`
- `energy_weight_pct = 30.0`

当 `eC/eG` 可用时，会构造能效比例：

$$
r_{energy}=\frac{P_{g,eff}\cdot e_C}{P_{c,eff}\cdot e_G + P_{g,eff}\cdot e_C}
$$

最终：

$$
r^*=\frac{w_p\cdot r^* + w_e\cdot r_{energy}}{w_p+w_e}
$$

若能耗数据不可用，则维持纯性能导向。

##### 3.13.9 解压路径额外惩罚

当 adaptive 用在解压（调用时 `input == NULL`）时，代码会施加 `dec_host_penalty_pct = 5.0`，把 `r_star` 乘以 `0.95`。

本质上这是“主机协调偏置”：

- 解压端合并/回放对主机侧更敏感；
- 因此默认让 GPU 比例略保守。

##### 3.13.10 连续比例到离散块：`hybrid_adaptive_adjust_gpu_blocks`

这一层是你反复点名的重点，因为真正执行的是块，不是浮点比例。

关键常量：

- `min_mixed = 8`（混合模式两侧最少块数）
- `quantum = 4`（块数量化步长）
- `collapse_small = 1`（小块数场景允许直接塌缩）

流程是：

1. `gpu_blocks = round(nblk * gpu_ratio)`；
2. 若 `nblk <= 2*min_mixed` 且是混合分配，则塌缩为全CPU或全GPU；
3. 否则保证两侧都不低于 `min_mixed`；
4. 对中间值按 `quantum` 量化；
5. 防止量化后越界到 `0` 或 `nblk`。

这一步直接消除“比例边界抖动 + 极小混合批次”的双重问题。

##### 3.13.11 压缩与解压两条调用链差异

压缩：

- `hybrid_compress_buf` 调 `choose_adaptive_gpu_ratio(..., input_buf, in_sz, ...)`；
- 有真实输入指针，不触发解压惩罚分支。

解压：

- `hybrid_decompress_buf` 调 `choose_adaptive_gpu_ratio(..., NULL, orig_sz, ...)`；
- 明确触发 `dec_host_penalty_pct` 的保守修正。

所以“同名 adaptive”在压缩/解压并非完全同值，而是共享框架、在解压有额外偏置。

##### 3.13.12 adaptive 输出如何进入 prefix 边界

调整后 GPU 块数会再变回比例：

$$
gpu\_ratio = \frac{adjusted\_gpu\_blocks}{nblk}
$$

随后进入 prefix 分割：

- GPU：`[0, gpu_count)`；
- CPU：`[gpu_count, nblk)`。

也就是说 adaptive 的最终产物不是任意索引集合，而是一个可解释、可复现的前缀边界。

#### 3.14 GPU pack kernel（实现级全展开）

这一节同样按真实代码路径展开，从 host 判定到 kernel 内部搬运，不再只讲“会打包”。

##### 3.14.1 pack 触发链（Host 侧）

`hybrid_compress_buf` 在 GPU 长度表回读后先计算：

1. `packed_total = sum(lengths[])`；
2. `sparse_bytes = gpu_count * worst_blk`；
3. 调 `hybrid_should_use_device_compaction(...)`。

判定门限：

- `pack_kernel` 必须存在；
- `num_blocks >= 8`；
- `packed_total < sparse_bytes`；
- 节省比例至少 `5%`。

##### 3.14.2 为什么 prefix 快路径下会强制关 pack

代码里有：

- `const int pack_zerocopy_prefix = 1;`
- 且 `if (pack_zerocopy_prefix && use_prefix_split) use_compaction = 0;`

原因是 prefix 场景可以直接 `clEnqueueReadBuffer` 一次回读连续前缀槽位，少一次 offsets 上传 + pack kernel 发射 + packed_out 回读。

这不是“pack 无效”，而是“在该路径下有更低开销的零拷近似路径”。

##### 3.14.3 开启 pack 后的 Host 端准备

当 `use_compaction` 为真时，Host 会准备两个设备缓冲：

1. `d_packed_off`：每块 packed 起始偏移；
2. `d_packed_out`：紧凑输出缓冲，总大小 `packed_total`。

offset 的构造通过映射写：

- `offset[0]=0`；
- 递增加 `lengths[i]`；
- 写入 `cl_uint` 数组。

这一步的正确性决定 pack kernel 是否会写重叠或越界。

##### 3.14.4 pack kernel 启动参数

Host 侧设置参数顺序：

1. `sparse_out` (`ws->d_out`)
2. `block_lens` (`ws->d_len`)
3. `packed_out` (`ws->d_packed_out`)
4. `packed_offsets` (`ws->d_packed_off`)
5. `worst_blk`
6. `total_blocks`

launch 几何：

- 初始 `pack_local=64`；
- 若块数更小，取不超过 `gpu_count` 的 2 的幂；
- `pack_global = gpu_count * pack_local`。

这与 kernel 内使用 `get_group_id(0)` 映射块号严格对应：一组处理一个块。

##### 3.14.5 kernel 内部并行模型

`lzo_pack_compressed_blocks`（`lzo1x.cl` 与 `lzo1y.cl` 同名实现）采用“块内并行搬运”：

1. `blk = get_group_id(0)` 决定当前块；
2. `lane = get_local_id(0)` 决定组内线程；
3. `src = sparse_out + blk * worst_blk`；
4. `dst = packed_out + packed_offsets[blk]`；
5. 实际拷贝长度由 `block_lens[blk]` 决定。

核心意义：块间完全独立，不需要跨块同步。

##### 3.14.6 小块路径（`len <= 32`）

当压缩块很小时，kernel 只让 `lane==0` 线程执行：

1. 先尝试 `uchar16` 向量搬运；
2. 再尝试 `uchar8`；
3. 尾部逐字节补齐。

这里的设计是：

- 小块不值得全组并行；
- 避免线程调度成本吞噬收益。

##### 3.14.7 大块路径（三段式）

当 `len > 32` 时走三段：

1. `vec32` 段：每次复制两个 `uchar16`（共 32 字节）；
2. `vec16` 段：处理剩余可整除 16 的部分；
3. scalar 段：处理最终尾字节。

每段都按 `lane` 条带化分工，步长是 `lanes * chunk`，保证组内线程均匀分配。

##### 3.14.8 正确性边界

pack kernel 的语义依赖两个前置不变量：

1. `block_lens[blk] <= worst_blk`；
2. `packed_offsets` 是严格前缀和。

只要这两点成立，`src` 与 `dst` 都是区间不重叠、边界可证的。

##### 3.14.9 pack 结果如何回写到 Host 缓冲

回读 `d_packed_out` 后有两种组装：

1. 有 `gpu_idx`（非连续映射）时：逐块按 `lengths[blk_idx]` 拷贝到 `out_buf + blk_idx * worst_blk`；
2. 无 `gpu_idx`（前缀连续）时：可直接连续 `memcpy`。

注意：即使走了 pack，最终 `out_buf` 仍按 `worst_blk` 槽位组织，便于后续统一容器写入路径。

##### 3.14.10 pack 与 timing 字段关系

pack 的时间统计并不是独立字段，而是并入：

- `pack_kernel_us` 累加到 `timing.gpu_kernel_us`；
- offsets 上传累加到 `timing.upload_us`；
- packed 回读仍计入 `timing.download_us`。

因此观察 pack 收益时，要同时看三个窗口，不能只盯 `gpu_kernel_us`。

##### 3.14.11 为什么 pack 有时“理论省字节，实测不提速”

常见原因：

1. `packed_total` 省得不够多，抵不过额外 kernel 与同步；
2. 输入块数少，`min_blocks=8` 附近的收益不稳定；
3. prefix 直读路径本来就很便宜，pack 反而多走一步。

这正是 `min_gain_pct=5` 与 prefix 直读优先策略存在的原因。

#### 3.15 adaptive 与 pack 的联合时序图（文字版）

下面把两者放到同一时序里，说明它们在一轮压缩中的先后关系：

1. 计算 `blk/nblk`；
2. adaptive 产出 `gpu_ratio`；
3. 块数修正产出 `gpu_count`；
4. CPU/GPU 并行压缩；
5. GPU 回读长度表；
6. 基于长度表判定是否 pack；
7. pack 开则走 offsets+pack kernel+packed 回读；
8. pack 关则走槽位直读；
9. 汇总 `lengths[]` 写容器。

关键结论：

- adaptive 决定“谁处理哪些块”；
- pack 决定“GPU结果怎么回读”；
- 两者作用域不同但串联在同一热路径上。

#### 3.16 代码映射补充（你点名关注的符号）

##### 3.16.1 adaptive 相关符号

| 关注点 | 代码符号 |
| --- | --- |
| 连续比例计算入口 | `choose_adaptive_gpu_ratio` |
| 块数离散化 | `hybrid_adaptive_adjust_gpu_blocks` |
| 样本索引策略 | `sampled_block_index` |
| 样本熵计算 | `lzo_calc_entropy` |
| CPU可用度 | `lzo_read_cpu_availability` |
| GPU可用度 | `lzo_read_gpu_availability` |
| 设备校准 | `lzo_calibrate_device_profile` |

##### 3.16.2 pack kernel 相关符号

| 关注点 | 代码符号 |
| --- | --- |
| 开关判定 | `hybrid_should_use_device_compaction` |
| 偏移缓冲 | `ws->d_packed_off` |
| 紧凑输出缓冲 | `ws->d_packed_out` |
| OpenCL kernel 名 | `lzo_pack_compressed_blocks` |
| 稀疏输入缓冲 | `ws->d_out` |
| 块长度缓冲 | `ws->d_len` |
| prefix 直读开关 | `pack_zerocopy_prefix` |

##### 3.16.3 你审代码时可直接盯的“关键常量”

| 常量 | 含义 |
| --- | --- |
| `min_mixed=8` | 混合模式最小块数门限 |
| `quantum=4` | adaptive 块数量化步长 |
| `min_blocks=8` | pack 启动最小块数 |
| `min_gain_pct=5` | pack 启动最小节省比例 |
| `dec_host_penalty_pct=5` | 解压 adaptive 保守惩罚 |
| `perf/energy=70/30` | adaptive 性能-能效默认权重 |

这张表的用途是让你在 code review 时快速验证“策略有没有被改动”，避免只看现象不看门限。

#### 3.17 当前采纳优化（主线一致性展开）

##### 3.17.1 严格压缩率守护主线（ratio guard v2）

- **动机**：先满足“Hybrid 压缩率不低于 CPU/GPU”的硬约束，再回收吞吐，避免只追速度导致 ratio 失真。
- **设计**：
  1. 在 adaptive 分流增加 strict ratio guard；
  2. 增加 ratio-aware 修正，抑制低比率极端选择；
  3. 修复非前缀索引映射场景的 GPU upload/readback 一致性。
- **实现**：
  - 关键文件：`/root/lzo-2.10/lzo_hybrid/lzo_hybrid_core.c`
  - 关键逻辑：ratio 守护分支 + 索引映射一致性修复。
- **效果**：
  - 工件：`/root/lzo-2.10/exp_results/runs/ratio_guard_ab_20260402_002534/lzo_hybrid_ab_v2_fullset.json`
  - 结果：`Comp +20.8622%/-7.7228%`，`Dec +6.7532%/-0.2692%`，`Ratio +0.244380/+0.565000 pctpt`
  - 结论：ratio 目标达成，作为后续优化锚点保留。

##### 3.17.2 有界 ratio 搜索（双算法约束）

- **动机**：adaptive 在 `lzo1x/lzo1y` 双算法上出现极端比例漂移，`Comp/Dec` 同时受损。
- **设计**：
  1. ratio refinement 改成有界搜索，禁止无边界漂移；
  2. 压缩与解压路径分别引入算法感知区间约束。
- **实现**：
  - 文件：`/root/lzo-2.10/lzo_hybrid/lzo_hybrid_core.c`、`/root/lzo-2.10/lzo_hybrid/lzo_hybrid.c`
  - 路径：`/root/lzo-2.10/exp_results/runs/deep_rework_subset_round3/runs/20260403_121754/`
- **效果**：
  - 相对 fixed `R=0.5`：
    - `lzo1x: Comp -18.29% -> -6.41%，Dec -16.26% -> -2.85%`
    - `lzo1y: Comp -19.72% -> -2.62%，Dec -7.20% -> -1.04%`
  - 结论：结构性回退显著收敛，为主线稳定化提供基础。

##### 3.17.3 降低 CPU-only 误触发 + adaptive ratio cache

- **动机**：小中型文件误触发 CPU-only 与 bench 循环重复 adaptive 求解共同拉低总吞吐。
- **设计**：
  1. 下调 `adaptive_skip_ocl_threshold`，减少 `AdaptiveGpuRatio=0`；
  2. 引入压缩/解压 ratio cache，命中后直接复用决策。
- **实现**：
  - 文件：`lzo_hybrid.c`、`lzo_hybrid_core.h`、`lzo_hybrid_core.c`
  - 运行目录：`/root/lzo-2.10/exp_results/runs/lzh_adaptive_deep_r1/runs/20260403_165528/`
- **效果**：
  - subset：`dComp +6.4270%/+6.1492%`，`dDec +3.0565%/+1.5811%`
  - fullset：`dComp +6.6477%/+6.3408%`，`dDec +5.3726%/+1.9963%`
  - `AdaptiveGpuRatio=0` 占比：`15% -> 0%`
  - 结论：subset/fullset 双通过，且直接命中主要回退来源。

---

### 4. 测试结果和分析

#### 4.1 测试方法与基线有效性

1. 样本：`/root/samples` 全集 50 文件，`Roundtrip_OK` 全通过。
2. strict 统一口径：`bench_seconds=3.5`，覆盖 CPU/GPU/HYBRID 全配置。
3. 主工件：
    - `/root/lzo-2.10/exp_results/baseline/fullset_current_strict/runs/20260404_merged/lzo_param_sweep.csv`
    - `sha256=3ff57c7656e99dcd2ca1c86640027aac41adecfa8e8763b1ab5af020caaf67aa`
4. 实现一致性：strict CSV 之后 `.c/.h/.cl` 新修改数为 0，数据与当前实现一一对应。
5. 结论口径：该 strict 基线是当前最新且主线最优（按已采纳策略集合）的评估锚点。

#### 4.2 按频率分解：CPU 引擎

CPU（按 `CF` 聚合）结果：

1. `CF=800MHz`：`CompTotal=1032.49`，`DecTotal=629.50 MB/s`，`Ratio=27.7751%`，`Power=6.97W`
2. `CF=1900MHz`：`CompTotal=2495.51`，`DecTotal=1531.79 MB/s`，`Ratio=27.7751%`，`Power=12.68W`
3. `CF=3000MHz`：`CompTotal=3829.63`，`DecTotal=2379.46 MB/s`，`Ratio=27.7751%`，`Power=30.47W`
4. `CF=5000MHz`：`CompTotal=4659.19`，`DecTotal=2944.92 MB/s`，`Ratio=27.7751%`，`Power=38.09W`

结论：CPU 吞吐随频率提升呈稳定增长，压缩率基本恒定，功耗高频段显著抬升。

#### 4.3 按频率分解：GPU 引擎

GPU（按 `GF` 聚合）结果：

1. `GF=500MHz`：`CompTotal=631.58`，`DecTotal=1084.77 MB/s`，`Ratio=26.9820%`，`CPU/GPU功耗=25.71/2.53W`
2. `GF=1000MHz`：`CompTotal=1237.82`，`DecTotal=2167.37 MB/s`，`Ratio=26.9820%`，`CPU/GPU功耗=26.02/5.72W`
3. `GF=1500MHz`：`CompTotal=1800.26`，`DecTotal=3223.11 MB/s`，`Ratio=26.9820%`，`CPU/GPU功耗=27.33/15.46W`

结论：GPU 压缩/解压吞吐均随频率近线性上升，压缩率稳定；高频点 GPU 功耗显著增加。

#### 4.4 按频率分解：HYBRID 引擎

HYBRID（按 `CF/GF` 频点对聚合）结果：

1. `CF/GF=800/500`：`CompTotal=909.69`，`DecTotal=718.55 MB/s`，`Ratio=27.3401%`，`CPU/GPU功耗=7.36/0.01W`
2. `CF/GF=800/1500`：`CompTotal=903.90`，`DecTotal=718.58 MB/s`，`Ratio=27.3402%`，`CPU/GPU功耗=7.36/0.03W`
3. `CF/GF=3000/500`：`CompTotal=3034.78`，`DecTotal=2592.13 MB/s`，`Ratio=27.3389%`，`CPU/GPU功耗=29.65/0.03W`
4. `CF/GF=3000/1500`：`CompTotal=3042.62`，`DecTotal=2575.87 MB/s`，`Ratio=27.3399%`，`CPU/GPU功耗=29.80/0.10W`
5. `CF/GF=5000/500`：`CompTotal=3339.48`，`DecTotal=2986.97 MB/s`，`Ratio=27.3381%`，`CPU/GPU功耗=34.34/0.03W`
6. `CF/GF=5000/1500`：`CompTotal=3349.36`，`DecTotal=2986.04 MB/s`，`Ratio=27.3395%`，`CPU/GPU功耗=34.32/0.10W`

结论：HYBRID 吞吐主要受 CPU 频率主导，GPU 频率带来增益但幅度较温和；压缩率在各频点稳定。

#### 4.5 功耗合理性确认（GPU 功耗低于 CPU）

在 strict 主工件中对 `Engine=GPU` 逐行检查条件 `CompGPUPower_W < CompCPUPower_W`：

1. 检查行数：`600`
2. 条件成立：`600/600`
3. 覆盖率：`100%`

结论：当前数据满足“GPU 功耗低于 CPU 功耗”的合理性要求。

#### 4.6 按文件对比（GPU/HYBRID 相对 CPU）

基于 `lzo_engine_vs_cpu_file_summary.csv`：

1. GPU vs CPU：
    - 压缩：`5` 升 / `45` 降，均值 `-46.37%`
    - 解压：`32` 升 / `18` 降，均值 `+42.03%`
    - 压缩率：均值 `-0.7931 pctpt`
2. HYBRID vs CPU：
    - 压缩：`35` 升 / `15` 降，均值 `+1.97%`
    - 解压：`43` 升 / `7` 降，均值 `+26.22%`
    - 压缩率：均值 `-0.4357 pctpt`

#### 4.7 Hybrid 内部：adaptive vs fixed(R=0.5)

按文件/频点/线程/算法配对后共 `1200` 对：

1. 总体：`dComp mean=-2.94%`，`median=-0.14%`；`dDec mean=-0.36%`，`median=-0.30%`；`dRatio mean=-0.0057 pctpt`
2. 胜场：`Comp 533/1200`，`Dec 557/1200`
3. 分算法：`lzo1x(dComp=-1.81%，dDec=+0.75%)`，`lzo1y(dComp=-4.06%，dDec=-1.46%)`

结论：adaptive 已接近 fixed，但 `lzo1y` 仍是主要负贡献方向，后续优化需继续针对该路径收敛。

---

### 5. 当前结论和后续方向

#### 5.1 当前结论

1. strict 基线已完成并落盘，`cpu/gpu/hybrid` 三份基线表可直接用于后续 A/B。
2. `lzo_hybrid` 在当前实现下仍应以 `fixed(R=0.5,prefix)` 作为默认稳态策略。
3. adaptive 在 LZO 线上已接近 fixed，但 `lzo1y` 方向仍是主要负贡献源。

#### 5.2 后续方向

1. 有效方向一：围绕 `lzo1y` 做 adaptive 稳定化（先压回退尾部，再追求均值提升）。
2. 有效方向二：引入更细粒度的“文件特征 + 频点”保护门限，减少极端文件（如 `x-ray`）大幅回退。
3. 有效方向三：保持 fullset strict 回归流程不变，防止“subset 看起来好、全量失效”。

#### 5.3 明确避开的无效方向

1. 在未处理 `lzo1y` 负贡献前，直接把 adaptive 设为全局默认。
2. 重新引入非 prefix 分区路径，增加语义复杂度并削弱可解释性。
3. 仅凭个别频点正向样本就下结论，不看全量文件分布。

#### 5.4 发布前核查

1. `partition_blocks_prefix` 是否仍为唯一路径；
2. 参数结构是否未引入额外分区字段；
3. 容器写入与解析字段是否一一对应；
4. fixed/adaptive 下 roundtrip 是否稳定通过；
5. current/baseline 哈希是否已刷新。

---

## Nvidia 平台（保留章节）

Nvidia 平台沿用同一容器定义与前缀分区语义。跨平台比较时需固定输入集、参数集与统计字段，避免把平台差异与口径差异叠加到同一结论中。

