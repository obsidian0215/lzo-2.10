# LZO Hybrid 实现：设计、优化与三引擎全量性能对比

更新时间：2026-03-09（测量语义二次修正后）

---

## 1. 项目概述

`lzo_hybrid` 是基于 `lzo_gpu` 和 `lzo_cpu` 的融合并行压缩/解压实现。核心思路：将文件的块（block）按可配置比例分配给 GPU（OpenCL 内核）和 CPU（pthreads + liblzo2）两路同时执行，通过并行流水线最大化端到端吞吐量。支持 lzo1x 和 lzo1y 两种算法。

### 1.1 硬件平台

| 组件 | 规格 |
|------|------|
| CPU | 13th Gen Intel Core i7-1370P |
| GPU | Intel Iris Xe Graphics (Gen12 LP), 96 EUs, 7 threads/EU, 64KB SLM |
| GPU 最大频率 | 1500 MHz |
| 内存架构 | 共享内存 (iGPU)，CPU 与 GPU 竞争内存带宽 |
| OpenCL | 3.0 NEO Driver |
| 能耗遥测 | Intel RAPL: `cpu=rapl:package-0`, `gpu=rapl:uncore` |

### 1.2 设计目标

1. **吞吐量最大化**：利用 CPU+GPU 并行流水线超越纯 GPU 和纯 CPU 的端到端吞吐量
2. **灵活配比**：`gpu_ratio` 参数控制 GPU/CPU 工作分配比例（0.0 = 纯 CPU, 1.0 = 纯 GPU）
3. **算法兼容性优先**：GPU 和 CPU 使用相同的 LZO 算法，产出互相兼容的块格式；但系统级压缩率仍可能因 hybrid 容器和分路策略而劣于纯 CPU/GPU
4. **能效对比**：在不同频率下评估三种引擎的能效特性

---

## 2. 整体架构

### 2.1 系统组件

```
lzo_hybrid/
├── lzo_hybrid.c           主入口：CLI 解析、OCL 初始化、内核加载、分发调度
├── lzo_hybrid_core.c      核心逻辑：缓冲区管理、CPU/GPU 并行分发、内存基准循环
├── lzo_hybrid_core.h      接口定义：hybrid_params_t, hybrid_workspace_t, hybrid_timing_t
├── Makefile               构建：gnu11 标准，链接 CPU LZO 库 + GPU 工具模块
├── lzo1x.cl -> ../lzo_gpu/lzo1x.cl    GPU 内核（符号链接）
├── lzo1y.cl -> ../lzo_gpu/lzo1y.cl    GPU 内核（符号链接）
└── *.clbin -> ../lzo_gpu/*.clbin      预编译内核缓存（符号链接）
```

### 2.2 数据流与并行调度

```
                    输入文件
                       │
               ┌───────┴───────┐
               │  块分割 (BS)  │
               └───────┬───────┘
                       │
          ┌────────────┴────────────┐
          │ gpu_count = nblk × R    │  R = gpu_ratio
          │ cpu_count = nblk - gpu  │
          └────┬───────────────┬────┘
               │               │
      ┌────────▼────────┐  ┌───▼──────────────┐
      │   GPU 路径      │  │    CPU 路径       │
      │ blocks[0..gpu-1]│  │ blocks[gpu..nblk-1]│
      │ OpenCL kernel   │  │ pthreads + liblzo2│
      │ 批量并行执行    │  │ 原子工作窃取      │
      └────────┬────────┘  └───┬──────────────┘
               │               │
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │  结果合并     │
               │  sequential   │
               └───────────────┘
```

**关键设计细节：**

1. **块分配策略**：GPU 获得前 `gpu_count` 个块，CPU 获得剩余块。GPU 块使用 OpenCL 内核并行处理（一个 work-item 处理一个块），CPU 块使用 pthreads 多线程处理。

2. **CPU 工作窃取**：CPU 线程使用原子计数器 (`__atomic_fetch_add`) 实现无锁工作窃取，每个线程动态取下一个待处理块。

3. **GPU 全局大小修正**：GPU 内核的 `global_size` 设置为 `gpu_count`（而非 `nblk`），确保 GPU 只处理分配给它的块。此修正带来解压吞吐量 +7.4% 提升。

4. **文件格式兼容**：CPU 和 GPU 使用相同的 MAGIC 0x4C5A 容器格式，块可互换。

### 2.3 内核端组件（GPU 路径）

GPU 内核直接复用 `lzo_gpu` 的实现，包含以下核心组件：

| 组件 | 功能 |
|------|------|
| **32-bit 字典** | `epoch_12 \| offset_20` 打包的紧凑哈希表 |
| **延迟字典写入** | 先批量读 4 个位置、检查匹配，再批量写入 |
| **匹配搜索** | 3 字节前缀哈希 → 4 位置线性探测 |
| **LZO 编码引擎** | 标准 LZO1X/LZO1Y 编码（literal run + match copy） |
| **解压引擎** | 状态机解码：literal/match/long-match/EOS 四状态 |

### 2.4 主机端组件

| 组件 | 功能 |
|------|------|
| **OpenCL 缓冲区管理** | grow-only 策略（只增不减），避免 CL buffer alloc/release 开销 |
| **非阻塞传输** | `clEnqueueWriteBuffer(non_blocking)` + `clFinish()` |
| **内核参数缓存** | 只在缓冲区增长时重设 kernel arg |
| **双基准模式** | `--bench` 保留历史内存循环语义；当前新增 `--bench-io` 作为正式 total-throughput 口径，在 warmed process 内执行真实 file-backed compress→write / read→decompress→write 循环 |

---

## 3. 开发与优化历史

### 3.1 初始实现

在 `lzo_gpu` 全量优化完成后（commit `fd406e7`）基础上创建。核心工作：

1. **并行框架搭建**：`hybrid_params_t` 添加 `gpu_ratio` 和 `cpu_threads` 参数，分发逻辑根据比例切分块到 GPU 和 CPU 路径。

2. **CPU 路径集成**：链接 `liblzo2`（系统安装），使用 `lzo1x_1_compress` / `lzo1y_1_compress` 及对应 `_decompress_safe` 函数。CPU 路径使用 pthreads 多线程 + 原子工作窃取。

3. **构建修复**：
   - `LZO1Y_1_MEM_COMPRESS` → `LZO1Y_MEM_COMPRESS`（liblzo2 API 差异）
   - `-std=c11` → `-std=gnu11`（需要 `pthreads` 和 `__atomic` 扩展）

### 3.2 内存基准优化

初始实现中 total_tp 远低于 kernel_tp（差距可达 50%），原因是 bench 循环包含文件 I/O 开销。

**第一阶段解决方案**：重写 bench 循环为纯内存操作——首次从文件读入缓冲区后，后续迭代直接操作内存中的数据。这个阶段性优化对剥离内核与主机端差异很有价值。

**当前最终口径补充**：在本轮 methodology correction 中，又新增了真正的 `--bench-io`，因为纯内存循环虽然适合做历史性能剖析，但不足以作为最终 system-level total throughput 的 ground truth。当前文档里的正式 total-throughput 结论一律以 warmed in-binary `--bench-io` 为准。

### 3.3 解压分发修复

发现 GPU 解压路径的 `global_size` 使用了 `nblk`（所有块数）而非 `gpu_count`（GPU 分配块数），导致 GPU 不必要地处理了 CPU 应处理的块。

**修复**：将 `clEnqueueNDRangeKernel` 的 `global_size` 从 `nblk` 改为 `gpu_count`。解压总吞吐量提升 +7.4%。

### 3.4 bench_lzo.py 集成

为基准测试框架 `bench_lzo.py` 添加完整的 hybrid 引擎支持：

- 新增常量：`LZO_HYBRID_BIN`, `HYBRID_BLOCK_SIZES=[16K, 32K, 64K]`, `HYBRID_GPU_RATIOS=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]`, `HYBRID_CPU_THREADS=[1, 2, 4]`
- 新增函数：`run_lzo_hybrid()` 解析 hybrid 二进制的输出格式
- 新增 CLI 参数：`--hybrid-only`, `--hybrid-block-sizes`, `--hybrid-gpu-ratios`, `--hybrid-cpu-threads`
- 三引擎标志逻辑：`run_cpu` / `run_gpu` / `run_hybrid` 独立控制

### 3.5 CPU total_tp 修复

原始 CPU 基准 (`lzo_cpu/lzo_frag.c`) 只输出 kernel_tp，缺少 total_tp。

**修复**：在 `run_stable_kernel_bench()` 函数中添加 wall-clock 计时（`clock_gettime(CLOCK_MONOTONIC)` 包裹 `compress_multi` / `decompress_multi` 调用），输出 `total_tp=X.XX MB/s`。实测 CPU total_tp 与 kernel_tp 非常接近（差异 < 0.1 MB/s）。

---

## 4. 全量实验设计（历史矩阵存档 + 当前 corrected run 入口）

### 4.1 测试矩阵

| 引擎 | 参数空间 | 配置数/文件 |
|------|---------|------------|
| CPU | 2 算法 × 3 线程(1,2,3) × 1 块大小(64K) | 6 |
| GPU | 2 算法 × 6 (BS×LSZ) 组合 × 1 Level | 12 |
| HYBRID | 2 算法 × 3 BS × 7 gpu_ratio × 3 cpu_threads | 126 |

下表保留的是 **第一轮全量 sweep 的历史矩阵**，用于说明 hybrid 参数空间是如何逐步铺开的；它对理解实现演进仍有价值，但**不再作为当前对外结论的证据来源**。

当前有效结论只引用：

- CPU/GPU/hybrid stitched artifact：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/lzo_param_sweep_merged.csv`
- full-corpus analysis bundle：`/root/analysis/20260309_full_refresh/lzo_best_per_file_medians.json`

历史 sweep 中每个配置曾在 3 个频率点（40%, 70%, 100%）测试，共 84 个测试文件（~7.3GB 总大小）。

| 数据集 | 行数 | 运行目录 |
|--------|------|---------|
| CPU | 1,512 | `exp_results/runs/20260306_015842/` |
| GPU | 3,024 | `exp_results/runs/20260305_191807/` (GPU 子集) |
| HYBRID | 31,752 | `exp_results/runs/20260306_031856/` |
| **总计** | **36,288** | `exp_results/full_comparison/merged_all.csv` |

### 4.2 测量指标

| 指标 | 单位 | 说明 |
|------|------|------|
| CompKernelMBs | MB/s | 压缩编解码吞吐量（由 `max(cpu_kernel_us, gpu_kernel_us)` 计算，表示 hybrid 下真实 codec 执行跨度，不含 file I/O 与纯 host coordination） |
| DecKernelMBs | MB/s | 解压编解码吞吐量（语义同上） |
| CompTotalMBs | MB/s | 压缩端到端总吞吐量（当前正式结论以 warmed in-binary `--bench-io` 的真实 file-backed wall-time 为准，包含运行时与文件路径开销，但不重复计入一次性冷启动） |
| DecTotalMBs | MB/s | 解压端到端总吞吐量（语义同上） |
| Ratio% | % | 压缩后大小/原始大小 × 100 |
| CompCPUEnergy_J | J | 压缩期间 CPU RAPL 能耗 |
| CompGPUEnergy_J | J | 压缩期间 GPU RAPL (uncore) 能耗 |

---

## 5. 修正后的全量实验结果（当前有效结论）

### 5.1 为什么必须重写这一节

旧版本第 5 节的主结论是：

- Hybrid 压缩 total throughput 100% 胜出；
- Hybrid 是三引擎综合最优。

这些结论已经与当前 corrected methodology 和当前 live 结果冲突，因此必须整体替换。

### 5.2 当前结果文件

- stitched artifact：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/lzo_param_sweep_merged.csv`
- provenance manifest：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/merge_manifest.json`

### 5.3 当前 best-per-engine medians

| Engine | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|
| CPU lzo1x | 615.68 | 642.44 | 2074.46 | 1765.48 | 21.68 | 14.74 |
| GPU lzo1x | **1334.63** | 808.86 | **5828.83** | **11600.56** | 23.38 | 13.85 |
| Hybrid fixed lzo1x | 925.46 | 821.24 | 2477.15 | 2292.82 | 21.68 | 12.28 |
| Hybrid adaptive lzo1x | 927.49 | **897.38** | 2580.50 | 2748.89 | 21.68 | 12.27 |
| CPU lzo1y | 613.45 | 642.66 | 1994.41 | 1756.91 | 22.12 | 14.80 |
| GPU lzo1y | 1330.73 | 808.09 | 5643.52 | 11587.72 | 23.19 | 14.15 |
| Hybrid fixed lzo1y | 926.01 | 812.25 | 2477.46 | 2256.97 | 22.12 | 12.26 |
| Hybrid adaptive lzo1y | **942.86** | **900.74** | **2643.72** | 2673.32 | 22.12 | 12.27 |

### 5.4 Winner counts

**Compression total throughput**：

- `lzo1x`：GPU 62，Hybrid fixed 17，Hybrid adaptive 4
- `lzo1y`：GPU 60，Hybrid fixed 18，Hybrid adaptive 5

**Decompression total throughput**：

- `lzo1x`：Hybrid adaptive 48，GPU 23，Hybrid fixed 12
- `lzo1y`：Hybrid adaptive 52，GPU 23，Hybrid fixed 8

### 5.5 Fixed vs adaptive

**Raw median across all hybrid rows**：

| Mode | Comp total MB/s | Dec total MB/s |
|---|---:|---:|
| lzo1x fixed | 748.03 | 745.72 |
| lzo1x adaptive | **906.62** | **833.92** |
| lzo1y fixed | 748.44 | 742.69 |
| lzo1y adaptive | **913.59** | **833.89** |

**Best-per-file median**：

| Mode | Comp total MB/s | Dec total MB/s |
|---|---:|---:|
| lzo1x fixed | 925.46 | 821.24 |
| lzo1x adaptive | **927.49** | **897.38** |
| lzo1y fixed | 926.01 | 812.25 |
| lzo1y adaptive | **942.86** | **900.74** |

### 5.6 当前正确解读

当前 corrected 结果的结论需要分两层理解：

1. **GPU 仍是 LZO family 的 compression 主路径**；
2. **此前 hybrid 被严重低估，原因是旧 Python harness 用额外 outer-process file-backed runs 覆盖了 binary 自己的 repeated-loop totals**；
3. **在 stitched 83-file full-corpus 结果中，adaptive 已不再弱于 fixed**；
4. **Adaptive 当前更准确地表现为：解压明显优于 fixed，并在部分模式下压缩也更强。**

因此旧文档中的 “Hybrid 在压缩 total 上 100% 胜出” 必须废弃；上一版把 hybrid 写得过度悲观的结论也必须废弃。

## 6. 最优配置推荐（按当前 corrected 结果重写）

### 6.1 当前默认推荐

如果目标是当前平台上的实际部署默认值，则推荐：

- **默认压缩引擎**：GPU-only
- **默认解压协同研究模式**：adaptive hybrid
- **理由**：当前结果已经从“GPU 全面统治”演变为“GPU 负责 compression，adaptive hybrid 在 decompression 上更强”

### 6.2 Hybrid 的当前合理定位

Hybrid 不再适合作为统一默认推荐路径，但仍适合作为：

- CPU/GPU 协同研究路径
- adaptive split 研究平台
- 某些未来更低开销调度器的基础实现

### 6.3 Adaptive 的推荐理解方式

当前 adaptive 的作用不是“让 hybrid 全面逆袭 GPU”，而是：

- 在 hybrid 自身内部提供另一种 split 选择；
- 在当前结果中，**decompression best-per-file 明显优于 fixed**，并且在 `lzo1x` / `lzo1y` 上的 compression 也不再弱于 fixed；
- 为后续更强调度策略保留真实实现入口。

## 7. 压缩率对比（按 corrected 结果更新）

当前 corrected best-per-engine medians：

| 引擎 | Ratio% |
|------|-------:|
| CPU lzo1x | 21.68 |
| GPU lzo1x | 23.38 |
| Hybrid fixed lzo1x | 21.68 |
| Hybrid adaptive lzo1x | 21.68 |
| CPU lzo1y | 22.12 |
| GPU lzo1y | 23.19 |
| Hybrid fixed lzo1y | 22.12 |
| Hybrid adaptive lzo1y | 22.12 |

这与旧文档“压缩率三者基本一致”的结论已经不同。

当前正确解释为：

- CPU 与 Hybrid 在同一算法线上的 ratio 已非常接近；
- GPU ratio 略高，但不构成灾难性代价；
- 这说明 LZO family 的 ratio trade-off 与 LZ4 family 不同，不能套用旧的统一描述。

## 7.1 CPU OpenCL 路径的当前定位

本轮还专门验证了“CPU 是否也应复用 OpenCL backend”的问题。当前代码通过 `FORCE_OPENCL_DEVICE=CPU` 已支持在 Intel CPU OpenCL 设备上运行同一套 OpenCL 路径，结果表明：

- **可运行且正确**；
- 在 `dickens`、`industrial_parent_0_pages_img.tar` 等代表性 case 上，CPU OpenCL 可以接近甚至短暂超过 GPU OpenCL；
- 但它没有稳定优于当前 native CPU path，也没有证明能把 hybrid 拉升到超过 GPU-only 的排序。

因此当前文档应把 CPU OpenCL 写成 **已验证的可行性/移植性结果**，而不是默认保留的主设计。

## 8. 当前结论

1. **LZO hybrid 的系统结构和实现细节仍然重要**：CPU/GPU 分路、atomic 工作窃取、OpenCL workspace、adaptive split 等都是真实存在且可复现的实现资产。  
2. **但当前 corrected 结果已经推翻旧结论**：Hybrid 不再是三引擎最优，更没有“100% 胜出”。同时，也不能再把 hybrid 描述成“几乎完全没有价值”的异常差路径。  
3. **GPU 是当前 LZO family 的 compression 主导引擎**：吞吐最高、active compression power 也更优。  
4. **Adaptive 已经不只是次要改善**：当前它在解压侧形成了明确优势，并在 fresh stitched artifact 上整体强于 fixed。  
5. **CPU OpenCL 虽然已确认可运行，但当前只适合作为 portability / research path**，不构成替代 native CPU path 或当前 hybrid 设计的依据。  
6. **LZO hybrid 当前最合理的定位是解压侧更强的研究型协同后端，而不是统一默认部署路径。**

简言之：

> `lzo_hybrid` 现在应被写成一个“结构完整、实现真实、调度经过 full-corpus stitched 验证后已显示出解压优势”的 CPU--GPU 协同系统；它不是新的统一默认引擎，但 adaptive 已经成为当前最值得继续优化的协同模式。

## 9. 2026-03-09 调度优化快照

本轮对 `lzo_hybrid` 做的关键实现更新包括：

- GPU/CPU block partition 从“GPU 前缀 + CPU 后缀”改为**跨文件分布式 block assignment**；
- 为分布式 GPU block assignment 增加了完整的 gather/scatter path：
  - GPU 压缩前先把 GPU-assigned blocks gather 成连续输入；
  - GPU 解压前先把 GPU-assigned compressed blocks gather 成连续压缩流；
  - 结果再 scatter 回全局 block 槽位；
- 修复了 gather staging buffer 过早释放导致的 GPU crash；
- 修复 `hybrid_bench()` 中 `--bench-io` 固定 `/tmp` 文件名问题。

### 9.1 subset 结果（优化后）

#### `dickens`，64K

| Engine / Mode | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---:|---:|---:|
| CPU-only | 171.80 | 214.27 | 61.91 |
| GPU-only | **465.19** | **703.11** | 63.56 |
| Hybrid fixed (0.7) | 374.56 | 336.27 | 63.06 |
| Hybrid adaptive | 202.44 | 246.58 | **62.31** |

#### `industrial_parent_0_pages_img.tar`，64K

| Mode | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---:|---:|---:|
| Hybrid fixed (0.7) | **925.81** | **874.05** | **17.53** |

### 9.2 当前结论

1. 分布式 GPU block assignment + gather/scatter 已在真实 roundtrip 和 warmed bench 中通过验证；
2. `lzo_hybrid` 当前已经不再被旧的前缀切分限制死在文件头局部模式上；
3. 但从当前 subset 结果看，**LZO GPU-only 仍然是默认最强引擎**，hybrid fixed 更像特定 workload 上的次优协同方案；
4. 因此本轮优化更准确的价值是：

> `lzo_hybrid` 的调度与 bench 路径已经被修正到可继续扩展的状态，但在当前 verified subset 上，它仍未形成对 `lzo_gpu` 的系统级反超。
