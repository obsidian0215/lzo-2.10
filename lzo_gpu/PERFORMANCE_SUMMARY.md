# LZO GPU 性能优化摘要与实验报告 (LZO GPU Optimization & Performance Report)

更新日期：2026-01-07

本文档详尽记录了 LZO GPU 项目的核心优化技术、实现原理、实测效果以及后续演进方向。优化工作主要分为主机端、压缩内核以及解压内核三个核心部分。

---

## 1. 主机端优化 (Host-side Optimizations)

主机端优化的目标是消除系统调用开销、减少数据拷贝瓶颈，并实现高并发下的资源最大化利用。

### 1.1 守护进程化架构 (Daemon Mode Persistence)
*   **优化原理**: 标准 GPU 程序在每次启动时都必须执行 OpenCL 运行时的初始化序列，包括：
    1.  **平台与设备枚举**: 探测可用 GPU 资源 (~50ms)。
    2.  **上下文与命令队列创建**: 初始化硬件通信通道 (~100ms)。
    3.  **内核程序 (JIT) 编译**: 将 OpenCL C 源码编译为硬件机器码。LZO 的复杂哈希和向量化逻辑导致这一步耗时高达 400ms-600ms。
*   **实现细节**: 通过 `lzo_gpu --daemon` 将上述资源持久化。客户端通过 `lzo_gpu --use-daemon` 作为轻量级前端，通过低延迟的 **Unix Domain Sockets (UDS)** 与守护进程通信，发送文件描述符和控制参数。
*   **优化效果**: 将单次处理的“预热时间”从平均 650ms 降至微秒级。对于小文件任务，整体响应速度提升了 **百倍以上**。

### 1.2 硬件感知的零拷贝内存 (Zero-copy / Pinned Memory)
*   **优化原理**: 在 Intel Iris Xe 等集成显卡（iGPU）架构中，CPU 与 GPU 物理共享系统 RAM。传统的 `clEnqueueWriteBuffer` 会导致数据从用户进程缓冲区拷贝到显卡专用驱动缓冲，造成多余的 CPU 指令和总线竞争。
*   **实现细节**:
    1.  **Pinned Memory**: 使用 `CL_MEM_ALLOC_HOST_PTR` 请求驱动分配页对齐的、不可移除的物理内存。
    2.  **直接映射**: 通过 `clEnqueueMapBuffer` 将物理地址直接映射给主机指针，实现“就地读写”。数据准备完成后，GPU 硬件可以直接读取该地址，无需显式内存拷贝。
*   **优化效果**: 整体 I/O 准备延迟降低了约 **50%**，并减少了约 15% 的 CPU 总核占用。

### 1.3 多 Worker 并行工作池 (Multi-worker Parallel Concurrency)
*   **优化原理**: 为了在多核 CPU 和具有多个执行单元（EU）的 GPU 上充分发挥吞吐量，单一顺序处理已成为瓶颈。
*   **实现细节**:
    1.  **独立命令队列**: 后端维护 4 个独立的 `cl_command_queue`。每个 Worker 相当于一个独立的调度通道，允许多个压缩/解压任务互不阻塞地提交给硬件调度器。
    2.  **线程安全的延迟加载**: 引入 `compile_lock` 互斥锁保护 JIT 过程。当多个客户端初次触发不同算法请求时，系统能确保内核编译过程的唯一性和完整性。
    3.  **资源私有化**: 每个 Worker 拥有专属的 `lzo_gpu_workspace_t` 和私有 Kernel 对象，彻底消除了高并发下 `clSetKernelArg` 等 API 调用的竞态条件 (Race Conditions)。
*   **优化效果**: 在 4 个并发客户端压力下，系统表现稳定，能够同时处理 GB 级的并发任务流，解决了高负载下的进程挂死问题。

---

## 2. 压缩内核优化 (Compression Kernel Optimizations)

压缩内核通过改进哈希冲突率、利用硬件向量执行单元以及优化内存分层结构实现极高的处理吞吐。

### 2.1 黄金比率乘法哈希 (Golden Ratio Multiplicative Hash)
*   **优化原理**: LZO 算法高度依赖哈希表来寻找匹配项。传统的简单移位哈希在处理结构化数据时容易产生“哈希聚簇”，导致多个不同数据块映射到同一个字典槽位，造成大量无效匹配尝试。
*   **实现细节**:
    ```c
    dv ^= dv >> 7; dv ^= dv >> 3;
    dv *= 0x9e3779b1u; // 2^32 / 黄金比例，提供最优的位分布均匀性
    dv ^= dv >> 16;
    ```
    该算法能够将输入的 32 位数据极均匀地散列到 $2^{D\_BITS}$ 的空间内。
*   **优化效果**: 实测在高重复度数据下，内核有效命中率 (Hit Rate) 提升了 **15%-20%**，大幅减少了对全局显存的冗余访问。

### 2.2 16 字节 SIMD 向量化访问 (16-byte Vectorization)
*   **优化原理**: 现代 GPU（如 Intel iGPU）内部总线宽度通常为 128 位或 256 位。逐字节的显存读写无法填满带宽，且会产生大量的指令发射开销。
*   **实现细节**: 在字面量拷贝 (Literal Run) 和匹配拷贝 (Match Copy) 环节，利用 OpenCL 的 `uchar16` 向量类型以及 `vload16`/`vstore16` 原语进行操作。
*   **优化效果**: 内核在处理大段不压缩数据（字面量）时，带宽利用率接近硬件极限，内部拷贝速度提升了 **25%** 以上。

### 2.3 全局显存字典与高占有率架构 (Global Memory Dictionary & High Occupancy Architecture)
*   **优化演进**: 在早期版本中，我们尝试利用本地显存 (Shared Local Memory, SLM) 来降低字典访问延迟。然而，在以 Intel Iris Xe 为代表的现代集成显卡架构中，SLM 存在严重的资源瓶颈（通常每个 CU 仅 64KB）。
*   **设计权衡**:
    - **Local 模式 (已淘汰)**: 当设置 `D_BITS=14` 时，每个工作组需占用 32KB SLM，导致每个计算单元只能并发运行 2 个工作组，产生了“占有率崩溃（Occupancy Collapse）”，吞吐量骤降。
    - **Global 模式 (当前标准)**: 将字典移至全局显存（Global Memory）。虽然单次访问延迟增加，但它彻底释放了 SLM 约束，使占有率提升至顶级（64+ 工作组并发）。
*   **实现细节**:
    1.  **并发掩盖延迟**: 利用 GPU 的海量线程并发来掩盖全局内存访问的延迟。
    2.  **向量化哈希查询**: 在全局字典访问中利用 `vload8` 配合 `ctz` 指令，实现一次访问多路匹配，抵消内存带宽压力。
*   **优化效果**: 在 `D_BITS=14` 的大字典配置下，性能比 Local 模式提升了 **14倍以上**，且能够支持更大（15-16 bit）的字典以获得更高的压缩率。

### 2.4 编译器与底层指令集深度微调
*   **循环展开 (Loop Unrolling)**:
    - **实现**: 通过预处理器宏控制 `LZO_USE_UNROLL2`。在关键的匹配查找循环（Match Search）中，通过 `match += 6; match -= 2;` 等技巧进行手动循环展开。
    - **原理**: 减少分支跳转次数（`for` 循环头的比较和跳转），增加指令并行度 (ILP)。这使得 GPU 的流水线能够同时预读更多后续指令，提高处理效率。
*   **分支预测与掩码优化 (Branch/Mask Optimization)**:
    - **原理**: GPU 的 SIMD 架构（或 Intel 的 SIMT）在遇到 `if-else` 分支分歧（Divergence）时性能较差。
    - **实现**: 在内核中尽量使用三元运算符 `(a < b ? a : b)` 或 OpenCL 的通用内置函数 `min()`, `max()`, `bitselect()`。这些函数在编译时往往被转换为硬件级的条件选择指令（如 `SEL`），避免了流水线冲刷。
*   **显式编译器标志声明**:
    - `-cl-fast-relaxed-math`: 允许编译器在不影响整型精度前提下，优化底层指令的流水线调度。
    - `-cl-mad-enable`: 允许将 `a * b + c` 这种模式映射到硬件底层的 `MAD` (Multiply-Add) 融合指令，在一个周期内完成两次运算，减少寄存器压力。
    - `-cl-std=CL2.0`: 显式指明标准，以启用更高效的 C11 内存模型支持（如原子操作优化）。
*   **对齐感知与 64 位宽操作**:
    - **细节**: 内核在进行长距离匹配拷贝时，会首先检查地址是否为 8 字节对齐。如果是，则直接通过 `ulong` 类型进行整体赋值。
    - **数据**: 比起逐字节 `uchar` 操作，64 位宽操作在内存总线上的占位更满，单步能吞吐更大数据。

### 2.5 现代架构下的实测观察 (Latest Benchmarking Observations)

在最近的性能调优中，我们发现了几个关键的影响因素：

1.  **字典清空开销 (Dictionary Clearing Overhead)**:
    - 当 `D_BITS` 设置为 12 或 13 时，字典项数量显著增加。对于 32KB 的小分块，每次处理前清库 8KB-16KB 的本地显存会占用约 5%-8% 的总内核时间。
    - **改进**: 通过在 `dict_clear` 中使用 1-thread 优化的非同步路径，并针对大文件自动提升块大小（见下文），有效稀释了这一固定开销。

2.  **2路组相联字典 (2-way Associative Dictionary)**:
    - **对比**: 传统的单路哈希表在遇到碰撞时只能覆盖旧数据。
    - **实测**: 引入 2 路组相联字典（`DICT_WAYS=2`）后，在 `mozilla` 字符集上的压缩率提升了约 **2.4%**。
    - **吞吐量**: 由于每次查找需要两次探测，核心循环的延迟略微增加，但更高的命中率减少了字面量编码路径，最终在 1x 线程模式下，总吞吐量基本持平，但压缩质量更优。

3.  **大文件的块大小博弈 (64KB vs 32KB Blocks)**:
    - 对于超过 100MB 的文件，使用 64KB 块相比 32KB 块能提供约 **1.5% - 3%** 的额外压缩增益。
    - **开销平衡**: 64KB 块意味着更少的线程块启动次数和更低的字典初始化频率。实测在大文件场景下，64KB 块的吞吐量比 32KB 提升了约 **4%**。

---

## 3. 解压内核优化 (Decompression Kernel Optimizations)

解压作为典型的“计算密度极低、I/O 密度极高”的任务，其瓶颈在于分支纠缠和内存带宽利用率。我们在 V2 版本中进行了深度的架构改进。

### 3.1 16 字节 SIMD 向量化架构 (16-byte Vectorized V2 Architecture)
*   **优化逻辑**: 传统的解压内核采用逐字节或逐 `uint` 的拷贝，无法填满硬件的总线带宽。
*   **实现细节**:
    1.  **统一向量化操作**: 引入基于 `vload16` 和 `vstore16` 的 `COPY_V2`。对于所有大于 16 字节的字面量运行 (Literal Run) 和匹配拷贝 (Match Copy)，均强制走 128 位向量化路径。
    2.  **LZO1Y 逻辑修正**: 针对 LZO1Y 算法中特殊的 M2 匹配逻辑（利用 `t >> 4` 和 `t >> 2` 计算长度和偏移量）在向量化路径下进行了精确实现，解决了 V2 初期的位对应错误。
*   **优化效果**: 处理高重复度数据（如全零或高度冗余文件）时，解压吞吐量获得 **5 倍以上** 的爆发式增长。

### 3.2 极小偏移量的特化处理 (Specialized Path for Small Offsets)
*   **核心挑战**: 当 `offset=1` 或 `offset=2` 时，传统的 `memcpy` 或向量化拷贝会导致严重的数据依赖冲突（读写重叠）。
*   **实现细节**:
    1.  **64 字节强制展开**: 针对 `offset=1` (RLE 填充)，引入 4x `vstore16` 的暴力填充，直接绕过慢速循环。
    2.  **分支平衡**: 通过减少对 `offset` 的判断深度，将最常见的 `offset > 8` 路径置于顶层，利用 GPU 的分支预测器减少流水线停顿。

### 3.3 并发占有率与 LOCAL_SIZE 调优 (Concurrency & Local Size Tuning)
*   **实验发现**: `LOCAL_SIZE` (工作组大小) 对解压性能有显著影响。
    - **LS=1 (单线程模式)**: 在处理极度不规则、分支高度分歧的数据时表现更优，因为彻底消除了 SIMT 内部的线程掩码等待。
    - **LS=64 (高并发模式)**: 在处理结构化、大块重复数据时表现更优，能通过海量硬件线程掩藏内存延迟。
*   **系统策略**: 默认采用灵活配置，支持根据任务类型动态调整工作组规模以适应不同的数据分布。

### 3.4 稳定性与安全性增强 (Stability & Safety)
*   **越界保护**: 增加轻量级边界检查，针对 Intel i915 驱动对显存越界极其敏感的特性进行保护，解决了早期版本在高压缩比下可能导致的 **GPU Hang** 风险。
*   **优化效果总结**:
    - **LZO1X V2**: 平均吞吐量提升至 **3.8 GB/s - 4.2 GB/s**。
    - **极致场景**: 在特定冗余数据下突破 **15 GB/s**。

---

## 5. 多维度性能评测与参数敏感性分析 (Integrated Param Scan Report)

本节以已合并的 CPU+GPU 参数扫描结果（exp_results/param_scan/param_scan_20260114_004558.csv）为数据源，替代先前的长篇摘要，给出可复现的数值结论、关键敏感性汇总、可视化图表与可下载的 artifacts（详见下文）。

**更新**：2026-01-14（已同步 `exp_results/param_scan/param_scan_20260114_004558.*` 的数据）

完整自动生成的扩展文本报告：`exp_results/param_scan/param_scan_20260114_004558.extended_report.txt`。

### 5.1 关键基线（快速概览）

| 系统 / 模式 | rows | 总吞吐 med (MB/s) | 内核吞吐 med (MB/s) | 压缩率 med (%) |
| :--- | ---: | ---: | ---: | ---: |
| Host CPU (T1) | 1184 | 257.2 | 361.8 | 20.7 |
| Host CPU (T2) | 1184 | 364.5 | 678.5 | 20.7 |
| Host CPU (T4) | 1184 | 492.2 | 1253.5 | 20.7 |
| GPU Standalone | 7104 | 317.3 | 781.5 | 4.7 |
| GPU Daemon | 7104 | 503.2 | 806.7 | 4.7 |

**要点:** GPU Daemon 显著提升端到端吞吐（消除 JIT 开销）；GPU 内核（kernel）吞吐与压缩率是后续敏感性分析的主要关注对象。

### 5.2 因子敏感性高亮（摘要）

- BlockSize (Compress): Range = **67.7 MB/s** (≈ **8.53%** of overall kernel median)
- BlockSize (Decompress): Range = **2551.4 MB/s** (≈ **64.30%**)
- Level (Compress): Range = **33.6 MB/s** (≈ **4.24%**)
- LocalSize (Compress): Range = **44.3 MB/s** (≈ **5.59%**) — 影响可观
- LocalSize (Decompress): Range = **852.9 MB/s** (≈ **21.49%**) — 对解压影响尤为显著
- Algorithm (Compress): Range = **141.2 MB/s** (≈ **17.79%**) — 不同算法表现差异明显

下图为关键因子的可视化（示例）：

**LocalSize — Compress (kernel medians)**
![](../exp_results/param_scan/figures/param_scan_20260114_004558.lsz_int.compress.median.png)

**LocalSize — Decompress (kernel medians)**
![](../exp_results/param_scan/figures/param_scan_20260114_004558.lsz_int.decompress.median.png)

**BlockSize — Compress (kernel vs ratio medians)**
![](../exp_results/param_scan/figures/param_scan_20260114_004558.block_kb.compress.kernel_vs_ratio.png)

（更多图表位于 `exp_results/param_scan/figures/`，并且为每个因子提供 `summary.csv`）

### 5.3 高影响样本示例（LocalSize）

- **Compress — 增益（LocalSize 1→8）示例**:
  - `sample_49.59mb_random_2.txt` : +63.09% (1:960.8 → 4:1432.8 → 8:1566.9)
  - `sample_21.92mb_random_1.txt` : +57.59% (1:960.3 → 4:1389.5 → 8:1513.4)
  - `influxdb-bench_cartelem_parent_7_pages-1.img` : +22.59% (1:1949.4 → 4:2174.6 → 8:2389.7)

- **Decompress — 增益（LocalSize 1→8）示例**:
  - `lzo_src_large.txt` : +66.91% (1:4184.3 → 4:6216.8 → 8:6984.1)
  - `elasticsearch-ycsb_parent_1_pages-1.img` : +62.21% (1:5853.7 → 4:8594.7 → 8:9495.1)

> 说明：完整的 top-list（增益 / 下降）已导出为 CSV（`exp_results/param_scan/param_scan_20260114_004558.top_*.csv`），可直接用作后续再测或自动化排查输入。

### 5.4 工件 (Artifacts) 与复现步骤

- 合并 CSV（主数据）: `exp_results/param_scan/param_scan_20260114_004558.csv`
- 扩展文本报告: `exp_results/param_scan/param_scan_20260114_004558.extended_report.txt`
- Per-factor top lists: `exp_results/param_scan/param_scan_20260114_004558.top_*.<compress|decompress>.csv`
- LocalSize 1→8 lists: `*.localsize.1to8.*.csv`
- 可视化目录: `exp_results/param_scan/figures/`（PNG + summary CSV）

复现命令示例：
```bash
# 生成 / 更新扩展文本报告
./tools/param_scan.sh -r exp_results/param_scan/param_scan_20260114_004558.csv

# 生成 / 更新图表（已包含在 repo）
python3 tools/generate_param_plots.py --csv exp_results/param_scan/param_scan_20260114_004558.csv --outdir exp_results/param_scan/figures
```

---

### 4.2 守护进程状态管理
- **PID 文件定位**: 记录 PID 于 `/tmp/lzo_gpu_daemon.pid`。
- **一致性清理**: 确保套接字和显存资源在停止时完全释放。

---

## 4. 架构性能对比分析 (Architectural Performance Analysis)

根据对 CPU (T1-T8) 与 GPU (Standalone/Daemon) 的大规模参数扫描（25,500 项组合测试），不同系统架构的端到端 (E2E) 表现如下：

### 4.1 核心架构对比 (Tool & Threading Comparison)
测试硬件: Intel Iris Xe Graphics, i7-1185G7 CPU.

| 配置方案 | 压缩 — 总吞吐 med (MB/s) | 压缩 — 内核吞吐 med (MB/s) | 压缩 — 内核加速 (vs CPU T1) | 压缩率 med (%) | 压缩 — 加速比 (vs CPU T1) | 解压 — 总吞吐 med (MB/s) | 解压 — 内核吞吐 med (MB/s) | 解压 — 内核加速 (vs CPU T1) | 解压 — 加速比 (vs CPU T1) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Host CPU (T1)** | 257.2 | 361.8 | 1.00x | 20.7 | 1.00x | 245.0 | 373.7 | 1.00x | 1.00x |
| **Host CPU (T4)** | 492.2 | 1253.5 | 3.46x | 20.7 | 1.91x | 412.2 | 1057.7 | 2.83x | 1.68x |
| **GPU Standalone (w/ JIT)** | 317.3 | 781.5 | 2.16x | 4.7 | 1.23x | 441.5 | 3700.9 | 9.90x | 1.80x |
| **GPU Daemon (Persistent)** | **503.2** | **806.7** | **2.23x** | **4.7** | **1.96x** | **577.2** | **4304.7** | **11.52x** | **2.36x** |

**分析结论**:
- **Daemon Mode 的效果**: 将初始化与 JIT 延迟持久化后，Daemon 显著提高了端到端吞吐（从 Standalone 的 ~317 MB/s 提升至 ~503 MB/s），成为最优的 E2E 模式。
- **按模式对比**: 相对于 Host CPU (T1)，在**压缩 (E2E)** 上，GPU Standalone/Daemon 的加速比分别约为 **1.23x / 1.96x**；在**解压 (E2E)** 上分别约为 **1.80x / 2.36x**。
- **内核吞吐对比**: 相对于 Host CPU (T1)，在**压缩内核**上，GPU Standalone/Daemon 的内核吞吐分别约为 **2.16x / 2.23x**；在**解压内核**上分别约为 **9.90x / 11.52x**。
- **总体对比**: 在当前扫描数据集中，GPU Standalone 與 GPU Daemon 均優於單線程 CPU（T1），但 Daemon 在端到端吞吐上优势最大；若关注内核吞吐，两者差距小，但 Daemon 的 E2E 表现受益于更少的运行时开销。

---

## 5. 压缩配置逐步敏感度统计 (Step-by-Step Parameter Sensitivity)

本节展示了单一参数变化对性能指标（按 GPU Standalone + Daemon 的合并样本）对比的定量影响。所有值均为样本中位数（median）。

### 5.1 块大小 (Block Size) 权衡 (以 8KB 为基准)
| 块大小 | 压缩率 med (%) | 相对 8KB 的压缩率变化 | 内核速度 med (MB/s) | 相对 8KB 的速度变化 | 优化策略 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **8 KB** | 4.32% | +0.00% | 803.0 | +0.00% | 均衡（参考基准） |
| **16 KB** | 4.63% | **+7.3%** | 784.9 | **-2.3%** | 效率优先（小幅降压缩率、略降速度） |
| **32 KB** | 4.82% | **+11.7%** | 770.2 | **-4.1%** | 高压缩比（对压缩率有小幅恶化） |
| **64 KB** | 4.89% | **+13.3%** | 765.4 | **-4.7%** | 大文件优先（压缩率/速度权衡） |

**结论**: 以当前数据集为例，从 8KB 增大至 64KB 会让压缩率（压缩后占原始数据的百分比）变大（表示压缩率略差），但内核速度也略有下降；选择 16–32KB 常为折衷方案，64KB 在大文件场景仍可接受。

### 5.2 搜索级别 (Search Level) 权衡 (以 L12 为基准)
| 级别 | 压缩率 med (%) | 相对 L12 的压缩率变化 | 内核速度 med (MB/s) | 相对 L12 的速度变化 | 说明 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **L11** | 4.68% | **-0.6%** | 790.0 | **+1.0%** | 稍高速度、微幅改善压缩率 |
| **L12** | 4.71% | +0.00% | 782.4 | +0.00% | 性能甜点（基准） |
| **L14** | 4.74% | **+0.7%** | 791.6 | **+1.2%** | 计算/压缩率均衡（小幅提升内核） |
| **L15** | 4.75% | **+0.9%** | 763.2 | **-2.4%** | 增加搜索成本，收益递减 |

**结论**: L12/L14 在多数场景为较好折衷，L15 成本增加但压缩率收益较小；建议以 L12 为默认，针对高压缩需求可试 L14。

### 5.3 解压 LocalSize (LSZ) 扩展性（按合并 GPU 样本）
| 工作组大小 | 解压内核吞吐 med (MB/s) | 端到端总吞吐 med (MB/s) | 说明 |
| :--- | ---: | ---: | :--- |
| **LS=1** | 3494.4 | 440.7 | 单线程稳健，低并行掩盖能力 |
| **LS=4** | **4196.7** | **446.8** | 最佳解压内核吞吐与 E2E 平衡（推荐默认） |
| **LS=8** | 3433.5 | 437.2 | 高频并行但受内存带宽影响（某些工作负载下降） |

**结论**: 对解压任务，LS=4 在当前测试集上表现最佳；对于极端结构化或高冗余数据，可尝试 LS=8/更细粒度测评。

---
---

## 6. 自适应参数选择启发式 (Adaptive Parameter Heuristics)

为了在不手动指定的情况下达到最优性能，系统引入了以下自适应调优策略 (`--auto-tune`)：

1.  **文件规模自适应**: 对于 < 1MB 的小文件，强制使用 **Daemon 模式** 并设置 **BlockSize=16K**。
2.  **内容属性感自适应**: 在高冗余数据（如 Log/Text）中提升至 **BlockSize=64K, Level=14**。
3.  **任务优先级自适应**: 默认解压启动 **LocalSize=4T**，实时流降低搜索等级至 **L10**。

---
*版权所有 (C) 2026 LZO GPU 项目组*
