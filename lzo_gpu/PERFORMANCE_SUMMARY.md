# LZO GPU 实现：设计、优化与性能分析

更新时间：2026-03-09

---

## 1. 项目概述

`lzo_gpu` 是基于 OpenCL 的 LZO 压缩/解压实现，目标是在 GPU 上实现超越 CPU 单线程及多线程（≤2 线程）的吞吐量，同时保持与 CPU 实现一致的压缩率。支持 lzo1x 和 lzo1y 两种算法。

### 1.1 硬件平台

| 组件 | 规格 |
|------|------|
| CPU | 13th Gen Intel Core i7-1370P |
| GPU | Intel Iris Xe Graphics (Gen12 LP), 96 EUs, 7 threads/EU, 64KB SLM |
| GPU 最大频率 | 1500 MHz |
| 内存架构 | 共享内存 (iGPU)，零拷贝可用 |
| OpenCL | 3.0 NEO Driver |
| 能耗遥测 | Intel RAPL: `cpu=rapl:package-0`, `gpu=rapl:uncore` |

### 1.2 设计目标

1. **吞吐量优先**：内核吞吐量和端到端总吞吐量均需超越 CPU（至少在 1-2 线程下）
2. **压缩率不退化**：与 CPU 实现保持一致的压缩率
3. **能效优势**：GPU 在内核执行期间的能耗应优于 CPU

---

## 2. 整体架构

### 2.1 系统组件

```
lzo_gpu/
├── lzo_gpu.c              主入口：CLI 解析、standalone 压缩/解压、bench 模式
├── lzo_gpu_core.c         核心主机端逻辑：OpenCL buffer 管理、块分割、数据传输、内核调度
├── lzo_gpu_core.h         核心接口：workspace、参数结构体、compress/decompress_core API
├── lzo_gpu_utils.c        OpenCL 工具：程序加载、clbin 缓存、自适应块大小选择、autotune
├── lzo_gpu_daemon.c       Daemon 模式：长驻进程，OpenCL 初始化一次，通过 Unix socket 服务
├── lzo_gpu_client.c       Client 模式：向 daemon 发送压缩/解压请求
├── lzo_gpu_protocol.h     Daemon 通信协议定义
├── lzo_defaults.h         默认参数：块大小、D_BITS、对齐、work-group 大小等
├── lzo_gpu.h              共享类型定义（主机与内核共用）
├── lzo1x.cl               lzo1x 算法 OpenCL 内核（压缩+解压）
├── lzo1y.cl               lzo1y 算法 OpenCL 内核（压缩+解压）
├── build_kernel.c         内核编译与 clbin 缓存管理
└── timing.h               高精度计时工具
```

### 2.2 数据流

```
输入文件 → 分块 → [零拷贝/标准拷贝] → GPU 全局内存
                                           ↓
                                    压缩/解压内核（每 work-item 处理一个块）
                                           ↓
                              结果读回 → 组装输出 → 写入文件
```

**压缩流程**：
1. 读取输入文件到内存（或零拷贝映射）
2. `choose_block_size()` 自适应确定块大小（考虑文件大小、熵、EU 数量）
3. 分块上传到 GPU 全局内存
4. 分配字典缓冲区（每 work-item 独立字典）
5. 启动压缩内核（每 work-item 独立处理一个块）
6. 读回压缩结果和长度
7. 组装 LZO 格式输出（magic + header + packed blocks）

**解压流程**：
1. 解析 LZO 文件头（magic、原始大小、块大小、块数量、长度表）
2. 打包压缩数据到连续缓冲区
3. 上传压缩数据 + 偏移表
4. 启动解压内核（每 work-item 独立解压一个块）
5. 读回解压结果

### 2.3 运行模式

| 模式 | 说明 | OpenCL 初始化 |
|------|------|---------------|
| Standalone | 单次压缩/解压，进程退出 | 每次启动初始化 |
| Daemon | 长驻进程，监听 Unix socket | 初始化一次，复用 |
| Client | 向 daemon 发请求 | 不初始化，复用 daemon |
| Bench | 性能测试模式，反复运行内核 | 初始化一次 |

### 2.4 I/O 模式

| 模式 | 机制 | 适用场景 |
|------|------|----------|
| 零拷贝 (默认) | `CL_MEM_ALLOC_HOST_PTR` + `fread` 直接写入映射指针 | iGPU 共享内存 |
| 标准拷贝 | 主机 buffer → `clEnqueueWriteBuffer` → 设备 buffer | dGPU / PCIe |

---

## 3. GPU 内核内部架构

### 3.1 压缩内核 (`lzo1x_block_compress` / `lzo1y_block_compress`)

每个 OpenCL work-item 独立处理一个数据块，核心组件：

#### 3.1.1 字典系统

**32-bit 紧凑字典**（当前实现）：
- 每个字典条目仅 4 字节：`epoch_12 | offset_20`
- `epoch`（12 位）：区分当前块与历史块的脏数据，替代清零操作
- `offset`（20 位）：块内偏移（覆盖 M4_MAX_OFFSET = 49151）
- D_BITS=12 时每 work-item 字典占 16KB；D_BITS=14 时占 64KB

```c
#define DICT_EPOCH_SHIFT 20
#define DICT_OFF_MASK    0x000FFFFFu

static inline void dict_store32(__global uint* dict, uint idx, uint offset, uint epoch) {
    dict[idx] = ((epoch & 0xFFFu) << DICT_EPOCH_SHIFT) | (offset & DICT_OFF_MASK);
}
static inline uint dict_load32(__global const uint* dict, uint idx, uint epoch, uint* valid) {
    uint entry = dict[idx];
    *valid = (((entry >> DICT_EPOCH_SHIFT) & 0xFFFu) == (epoch & 0xFFFu));
    return entry & DICT_OFF_MASK;
}
```

#### 3.1.2 哈希探测与延迟写入

**4 位置批量探测**：
- 从当前输入位置连续取 4 个位置的哈希索引 `idx_a = (uint4)(DINDEX(...),...)`
- 批量读取 4 个字典条目
- 检查所有 4 个位置的匹配
- **延迟写入（Deferred Store）**：读取和匹配检查完成后，才批量写回字典

这种分离读/写的模式改善了 GPU 内存控制器的调度效率。

#### 3.1.3 匹配编码

遵循 LZO 标准编码格式：
- **M2**：匹配长度 3-8，偏移 1-1024
- **M3**：匹配长度 3-33，偏移 1-16384
- **M4**：匹配长度 3+，偏移 16384-49151
- 字面量通过 run-length 编码

#### 3.1.4 Match 扩展与循环展开

```
LZO_USE_UNROLL2 = 1
```
在 match 长度扩展（比较当前位置与字典回溯位置的连续字节）时启用 2x 循环展开，减少循环控制开销并提高 ILP（指令级并行）。

### 3.2 解压内核 (`lzo1x_block_decompress` / `lzo1y_block_decompress`)

#### 3.2.1 命令解析

逐字节解析 LZO 压缩流的控制字节，识别字面量拷贝和匹配回拷指令。

#### 3.2.2 向量化拷贝

```c
static inline void LZO_MEMOPS_COPYN_FAST(void *dd, const void *ss, uint nn) {
    // 32 字节块 (uchar16 × 2)
    // 16 字节块 (uchar16)
    // 8 字节块 (uchar8)
    // 4 字节块 (uint)
    // 逐字节收尾
}
```

对字面量拷贝和非重叠匹配拷贝使用向量化加速。

#### 3.2.3 匹配拷贝优化

- **非重叠快路径**：当 `match_offset >= match_length` 时，直接使用向量化 `COPYN_FAST`
- **重叠安全路径**：当匹配偏移小于长度时（如 RLE 模式），使用 `COPY_MATCH` 逐字节处理以保证正确性

### 3.3 lzo1x vs lzo1y 差异

两种算法共享相同的内核架构和优化，区别仅在于 LZO 编码格式的位域定义（偏移/长度的编码位分配不同）。内核文件 `lzo1x.cl` 和 `lzo1y.cl` 结构完全对称。

---

## 4. 设计演进历史

### 4.1 时间线

基于 git 提交历史，按时间顺序列出关键设计变更：

| 日期 | 提交 | 变更 |
|------|------|------|
| 2025-10-28 | `177d630` | 初始 lzo_gpu 和 lzo_cpu 实现 |
| 2025-11-12 | `aa01cc3` | CLI 重构 |
| 2025-11-19 | `0f458f4` | 重构 lzo_gpu，添加参数扫描脚本 |
| 2025-11-19 | `ee6d4ef` | 移除 delayed store 和 usehost 模式 |
| 2025-11-21 | `3fbab57` | 移除 atomic 操作，避免 usehost |
| 2025-11-23 | `a2d2ed2` | 添加 Daemon 模式，GPU 优化 |
| 2025-11-24 | `9647cb3` | 优化 lzo1x 压缩/解压，自适应块大小选择 |
| 2025-11-25 | `491405c` | 修复 daemon 压缩，实现全路径零拷贝 |
| 2025-11-25 | `80ecbf1` | Standalone 模式独立化 |
| 2025-11-25 | `24fbdae` | 添加多线程 I/O |
| 2025-12-02 | `10ede4c` | 实现 wildcopy / timing，新增 benchmark 脚本 |
| 2025-12-05 | `b53f086` | 修复向量化解压 |
| 2025-12-05 | `4a38696` | 修复并对比 lzo_gpu 与 lzo_cpu |
| 2025-12-07 | `824ee11` | 首次微基准测试 GPU vs CPU |
| 2025-12-11 | `9f99457` | 计时精度修正 |
| 2025-12-20 | `7096f89` | 添加 lzo1y 支持，哈希表可配置化 |
| 2025-12-25 | `6118a51` | 架构重构：拆分 lzo_gpu_io/lzo_gpu_core，重写 standalone/daemon，实现解压零拷贝上传，添加 --level 和 --kernel-opt |
| 2026-01-14 | `99d8472` | 主机端与内核性能优化 |
| 2026-01-23 | `5019c3f` | 完成一个完整版本 |
| 2026-02-13 | `e0db950` | 确定 benchmark 语义，编写性能总结 |
| 2026-02-24 | `adf15e1` | 压缩路径优化基线快照 |
| 2026-02-25 | `006dde8` | 压缩匹配搜索探测向量宽度缩减 |
| 2026-02-26 | `0a2d593` | 解压匹配拷贝添加非重叠快路径 |
| 2026-02-26 | `321f9c1` | 解压拷贝热路径精简 |
| 2026-02-28 | `dd89965` | 启用 unroll2，增强 OpenCL 初始化鲁棒性 |
| 2026-03-05 | `b8a6533` | 内核与工具链改进 |
| 2026-03-05 | (未提交) | 32-bit 字典、延迟写入、主机端 bench 循环优化、总吞吐量测量 |

### 4.2 关键架构决策

**早期探索与淘汰（2025-11）**：
- **Atomic 操作**：尝试使用 GPU atomic 进行输出同步 → 性能差，移除
- **UseHost 模式**：`CL_MEM_USE_HOST_PTR` → 在 iGPU 上无优势，移除
- **Delayed Store**：早期延迟字典写入实验 → 当时移除（后以不同形式重新引入）

**架构稳定化（2025-12）**：
- 确立「每 work-item 处理一个块」的并行模型
- 实现自适应块大小选择（基于文件大小、熵、EU 数量）
- 向量化解压拷贝路径

**性能优化阶段（2026-02 ~ 03）**：
- 匹配搜索路径优化（探测宽度调整、循环展开）
- 解压路径优化（非重叠快路径、拷贝热路径精简）
- OpenCL 调度参数调优（WI_PER_CU 提升至 384）

---

## 5. 优化实现详述

### 5.1 内核端优化

#### 5.1.1 32-bit 紧凑字典（+15.3% 压缩吞吐量）

**问题**：原 64-bit 字典条目（`epoch_32 | entry_32`）导致每 work-item 字典占用过大。D_BITS=14 时每 WI 需要 128KB 字典内存，造成 GPU 全局内存带宽瓶颈。

**方案**：将字典条目压缩为 32-bit：`epoch_12 | offset_20`。
- 12-bit epoch（max 4095）足够区分块间脏数据
- 20-bit offset 覆盖 M4_MAX_OFFSET（49151）
- 字典内存减半：D_BITS=14 从 128KB → 64KB/WI

**结果**：
- 压缩内核吞吐量：平均 **+15.3%**（57/60 配置提升 >1%）
- 解压内核吞吐量：平均 **+5.7%**
- 压缩率变化：**0.00pp**

**修改文件**：`lzo1x.cl`, `lzo1y.cl`, `lzo_gpu_core.c`

#### 5.1.2 延迟字典写入（Deferred Dict Store）（+1.78% 压缩吞吐量）

**问题**：在 4 位置批量探测中，每次读取后立即写入字典，导致读写交替访问全局内存。

**方案**：分离读写操作——先批量读取所有 4 个字典条目并检查匹配，匹配处理完成后再批量写回所有字典更新。

**原理**：GPU 内存控制器（尤其在 iGPU 共享内存架构下）对连续读或连续写的调度效率高于读写交替。

**结果**：
- 压缩内核吞吐量：平均 **+1.78%**（9/12 配置提升 >1%，0/12 回退）
- lzo1x: +1.14%, lzo1y: +2.42%
- 压缩率变化：**+0.03pp**（可忽略）

**修改文件**：`lzo1x.cl`, `lzo1y.cl`

#### 5.1.3 匹配扩展循环展开（Unroll2）

**问题**：match 长度扩展循环每次迭代仅比较 1 字节，循环控制开销占比高。

**方案**：`LZO_USE_UNROLL2=1`，在 match 扩展热路径中 2x 展开。

**修改文件**：`lzo1x.cl`, `lzo1y.cl`

#### 5.1.4 解压非重叠快路径

**问题**：所有匹配拷贝都使用逐字节安全路径，即使大部分匹配的偏移远大于长度。

**方案**：当 `match_offset >= match_length` 时，直接使用向量化 `COPYN_FAST` 替代逐字节 `COPY_MATCH`。

**修改文件**：`lzo1x.cl`, `lzo1y.cl`

#### 5.1.5 探测向量宽度调整

将压缩内匹配搜索的批量探测从更宽向量缩减到 4 位置。经测试 8 位置批量探测导致约 10pp 压缩率退化，违反压缩率不退化约束。

#### 5.1.6 已评估但拒绝的内核优化

| 候选优化 | 原因 | 结论 |
|----------|------|------|
| Split-Dictionary（双数组） | 在 iGPU 共享内存上，1 次 8B 读 < 2 次 4B 读 | 1-5% 回退，拒绝 |
| 8 位置批量探测 | 探测 8 位置导致字典污染 | ~10pp 压缩率退化，拒绝 |
| Register Victim Cache (N=4) | iGPU 的 L3 cache 已天然承担此角色 | 无收益，拒绝 |
| Branchless Match 编码 | M2/M3/M4 已使用 uint 谓词，无 SIMD 发散 | 无收益，拒绝 |
| 解压内核宽化 | 解压 kernel_tp 已达 12-17 GB/s，瓶颈在主机端 | 无收益，拒绝 |

### 5.2 主机端优化

#### 5.2.1 解压 CL 缓冲区复用（+50.1% 解压内核吞吐量）

**问题**：Bench 模式下每次迭代的解压路径都重新 `clCreateBuffer` / `clReleaseMemObject`。在 iGPU 共享内存架构下，这导致严重的内存分配器争用，甚至拖慢内核执行。

**方案**：将 `d_comp`、`d_off`、`d_out`、`d_out_lens` 四个 CL 缓冲区移到循环外预分配，仅在容量不足时 grow-only 扩展。

**结果**：
- 解压内核吞吐量：**+50.1%**（40-57% 范围）
- 解压总吞吐量：**+20.4%**（16-24% 范围）
- 压缩性能：不变（预期内）

**修改文件**：`lzo_gpu.c` (`run_lzo_bench`)

#### 5.2.2 主机数组预分配

`h_lens`、`h_off`、`h_out_lens`、`packed` 等主机端数组在循环外一次分配，跨迭代复用，仅在需要时扩容。

**修改文件**：`lzo_gpu.c`

#### 5.2.3 非阻塞 CL 写入

将 `clEnqueueWriteBuffer` 的阻塞参数从 `CL_TRUE` 改为 `CL_FALSE`，在内核启动前统一 `clFinish()` 同步。这允许多个写入操作在命令队列中流水线执行。

**修改文件**：`lzo_gpu.c`

#### 5.2.4 内核参数缓存

解压内核的参数绑定和 dispatch 大小仅在缓冲区重新分配时更新，避免每次迭代重复调用 `clSetKernelArg`。

**修改文件**：`lzo_gpu.c`

#### 5.2.5 总吞吐量测量管线

在 bench 模式中同时记录内核吞吐量（纯 GPU 执行时间）和总吞吐量（包含数据传输和主机开销），为端到端性能评估提供完整数据。

**修改文件**：`lzo_gpu.c`, `bench_lzo.py`

### 5.3 调度与配置优化

#### 5.3.1 WI_PER_CU 提升

默认并行度从 256 提升到 384 work-items/CU，提高 CU 饱和度。

#### 5.3.2 OpenCL 初始化回退链

设备选择升级为 `GPU → DEFAULT → ALL` 三级回退，增强跨平台鲁棒性。

#### 5.3.3 显式设备选择与 CPU OpenCL 可行性验证

当前主程序已加入 `FORCE_OPENCL_DEVICE=CPU|GPU|DEFAULT|ALL`：

- 可显式绑定 Intel CPU OpenCL 设备或 Intel GPU 设备；
- 便于用同一套 OpenCL backend 验证 CPU OpenCL 的 correctness 与 steady-state 表现；
- 也为 hybrid / GPU 两条路径提供统一的设备选择语义。

fresh subset 验证表明：CPU OpenCL **功能上可运行**，并且在部分 case 上能接近 GPU OpenCL；但它并没有稳定展现出足够明确的系统级优势，去替代当前 native CPU baseline。因此当前更准确的定位是：

> **CPU OpenCL 是已验证的 portability path，而不是当前 Intel 平台上的默认 CPU 实现。**

#### 5.3.4 自适应块大小

`choose_block_size()` 综合考虑文件大小、采样熵（64KB 窗口）、EU 数量，动态选择块大小（4KB ~ 128KB），在并行度和单块压缩效率间取平衡。

---

## 6. 修正后的 CPU vs GPU 对比实验（当前有效基线）

### 6.1 为什么本节必须更新

旧版本第 6 节中的 CPU vs GPU 对比来自早期全量 sweep 与当时的测量语义；其中部分 total-throughput 结论已经被本轮 corrected methodology 更新。因此，本节现只保留 **当前对 CPU vs GPU 最终判断真正有效的 corrected baseline**。

### 6.2 当前实验设置

| 参数 | 当前有效设置 |
|------|-------------|
| 基线结果文件 | `/root/lzo-2.10/exp_results/runs/20260307_221942/lzo_param_sweep.csv` |
| 文件集 | 36 个已验证文件（来自 `/root/samples` 的当前 corrected base run 覆盖集） |
| 频率点 | 100% |
| 指标 | `CompKernelMBs`, `DecKernelMBs`, `CompTotalMBs`, `DecTotalMBs`, `Ratio%`, `CompCPUPower_W`, `CompGPUPower_W` |
| 正确性 | 仅统计 `Roundtrip_OK=yes` 结果 |

### 6.3 当前可信汇总方式

当前采用：

1. 对每个文件、每个 engine（CPU/GPU）在自己的配置空间中选最佳 `CompTotalMBs`；
2. 再对所有文件做 best-per-file median；
3. kernel throughput 用来解释 headroom，**不作为最终比较结论的主指标**。

### 6.4 当前 best-per-engine medians

| Engine | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 824.76 | 690.77 | 3149.84 | 1667.99 | 13.93 | 14.12 |
| **GPU** | **1501.12** | **811.37** | **9242.84** | **16692.40** | 13.95 | **13.67** |

### 6.5 当前结果分析

#### 6.5.1 total throughput

- compression total: GPU / CPU = **1.82x**
- decompression total: GPU / CPU = **1.17x**

这说明 corrected 之后，LZO GPU 仍然是 LZO family 中最强的默认引擎。

#### 6.5.2 ratio

当前 GPU ratio = 13.95%，CPU ratio = 13.93%，几乎等价。也就是说，GPU 的加速不是通过显著牺牲压缩率获得的。

#### 6.5.3 power

当前 active compression power：

- CPU = 14.12W
- GPU = 13.67W

因此当前结果不应描述为“GPU 更快但更耗电”，而应描述为：

> **GPU 在更高 steady-state total throughput 的同时，active compression power 还略低于 CPU。**

#### 6.5.4 kernel vs total

当前 GPU best-per-file medians：

- compression: 9242.84 MB/s kernel vs 1501.12 MB/s total
- decompression: 16692.40 MB/s kernel vs 811.37 MB/s total

这说明 LZO GPU 当前仍然是典型的“kernel 很强，但系统交付仍受 runtime/file-backed path 约束”的后端。因此 total throughput 才是决定系统结论的主指标。

### 6.6 当前应保留的最终结论

在 corrected steady-state total semantics 下：

1. **GPU 仍然明确优于 CPU**；
2. **压缩率几乎不变**；
3. **active compression power 不升反降**；
4. 因此 `lzo_gpu` 依然是当前项目中最强的 GPU 路径之一。

### 6.7 与 hybrid 和 CPU OpenCL 的关系

当前 LZO family 的关系应表述为：

- **GPU-only 仍是默认最佳引擎**；
- **Hybrid 在 corrected methodology 下已被重新评估为“次优但非异常差”**，说明之前的极端悲观印象部分来自测量口径问题；
- **CPU OpenCL 已验证可运行，但没有形成稳定优于 native CPU path 的证据**。

这意味着 `lzo_gpu` 的当前结论不是孤立成立的，而是在完成了 hybrid 重测和 CPU OpenCL 可行性验证之后，仍然保持为 LZO family 的主基线。

## 7. GPU-Only 优化阶段结果回顾（作为实现演进参考）

本节保留 GPU-only 优化阶段的中间实验结论，用于说明当前实现是如何收敛到现在这条主路径的。**这些数据不再作为最终 CPU vs GPU 结论的主依据**，但它们对理解实现演进仍然重要。

### 7.1 关键阶段性观察

历史优化阶段已经反复证明以下规律：

1. **更大的块大小通常更利于 total throughput 与 ratio**；
2. **D_BITS=12/14 之间存在 throughput vs ratio 的细微权衡**；
3. **iGPU 共享内存下，主机端 CL buffer 生命周期管理是决定 total throughput 的核心因素之一**；
4. **低熵数据会把 GPU 解压推到极高的 kernel throughput 区间，但这不等价于系统 total throughput 可以等比例放大**。

### 7.2 为什么这一节仍然值得保留

用户最关心的不只是“最终哪个数字最大”，还包括：

- 当前系统结构为什么会长成这样；
- 哪些优化真正起作用；
- 哪些方向虽然做过但最后被证明无效。

因此，本节作为 **优化演进档案** 继续保留，但所有最终对外比较结论均应以后续 corrected 第 6 节为准。

## 8. 当前结论

1. **LZO GPU 的架构与实现已经成熟**：包含 standalone / daemon / client、内核与主机端 runtime、workspace/buffer 复用、向量化解压路径等完整组件。
2. **历史优化路径仍然重要**：32-bit 紧凑字典、延迟写入、主机端 buffer 复用等，是当前结果成立的关键前提。
3. **当前 corrected baseline 证明 GPU 是 LZO family 的主导引擎**：在 steady-state total throughput 下压缩 1.82x 于 CPU、解压 1.17x 于 CPU。
4. **ratio 与 power 都没有破坏这个结论**：GPU 几乎不牺牲压缩率，且 active compression power 略低于 CPU。
5. **CPU OpenCL 当前不应被写成默认推荐设计**：它验证了统一 OpenCL backend 的可行性，但在本轮结果里更适合作为 portability/fallback path。
6. **跨家族比较需谨慎**：当前可以确认 `lzo_gpu` 是强结果，但若要严格断言优于 `lz4_gpu`，仍应使用 matched-corpus 对照。

简言之：

> 当前 `lzo_gpu` 是一条既有完整系统结构、又经过历史优化收敛、并且在 corrected steady-state total semantics 下仍然强势的正式主路径。

## 9. 2026-03-09 块大小敏感性快照

针对用户提出的“LZO GPU 对块大小似乎没有 LZ4 GPU 那么敏感”的问题，本轮重新做了 subset 验证。

### 9.1 `dickens`

| Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---:|---:|---:|
| 16K | 543.61 | **838.95** | 67.56 |
| 64K | **582.21** | 788.69 | **63.56** |

### 9.2 `industrial_parent_0_pages_img.tar`

| Block | Comp total MB/s | Dec total MB/s | Ratio % |
|---|---:|---:|---:|
| 16K | 976.49 | 1259.42 | 18.46 |
| 64K | **1073.01** | **1362.72** | **17.52** |

### 9.3 当前解释

1. LZO GPU 在当前 Intel iGPU 平台上仍然表现出**更弱的块大小敏感性**；
2. 64K 通常不会像此前 LZ4 某些异常 case 那样出现明显吞吐崩塌，反而在 total throughput 与 ratio 上略优；
3. 这与 LZO GPU 当前更稳定的 batched dictionary probing / copy 路径和成熟的 host runtime 设计相一致。
