# LZO GPU — 优化路线、状态与下一步计划

最后更新：2025-11-25

此文档仅包含路线与完成情况（短清单、优先级、下步行动），与 `PERFORMANCE_SUMMARY.md` 配合使用（后者描述具体实现与性能对比）。

---

## 当前总体状态（已完成）

已完成的重要阶段：

- Phase 1 — 并行度优化 (OCC_FACTOR、块大小调整)：已完成 ✅
- Phase 2 — 地址空间统一 (`__generic`)：已完成 ✅
- Phase 3 — 哈希函数优化（XOR混合）：已完成 ✅
- Phase 4 — 向量化匹配与向量化解压：已完成 ✅
- Phase 6.1 / Phase 8 — Pinned Memory + Zero-Copy I/O：已完成 ✅
- Phase 7.2 — 自适应块大小（熵驱动）：已完成 ✅

当前最佳内核变体（实测）：`lzo1x_1l`（压缩/解压平衡最佳）

## 关键当前瓶颈（按优先级）

1. Kernel 执行时间（大约占总时间 35~40%） — 需要内核微优化与缓存策略（Local Memory）
2. CPU 端处理与 I/O 管道（仍占大量时间，尤其是 Standalone） — 采用多线程/异步 I/O 可显著改善
3. 部分平台上的 Zero-Copy 对随机历史访问（尤其解压）有劣化风险 — 需要平台适配或混合策略

---

## 近期目标（最高优先级）

1) Phase 8.1 — 多线程 I/O（立刻优先）
   - 目标：把 CPU I/O、熵计算、文件写入与 GPU 传输/执行并行化。
   - 关键收益：减少 CPU 阻塞、隐藏 I/O 延迟，预期总吞吐 +15-20%。

2) Phase 10 — 异步传输流水线（短期并行进行）
   - 目标：双缓冲 / 多流把上传/执行/下载并行化。
   - 关键收益：隐藏传输时间，预期 +8-10%。

3) Phase 8.3 / 内核微调（并行）
   - 目标：减少寄存器压力、内存访问对齐、分支重排。
   - 关键收益：内核时延下降 3-8%。

---

## 中期（3-6个月）

- Phase 5 — Local Memory 字典缓存（如果 platform 支持）
  - 目标：把热点字典区域缓存到 LDS, 减少全局内存访问延迟
  - 风险：可用空间和并发更新策略需调优

- Phase 11 — ML 辅助决策（可选）
  - 目标：用轻量模型选择最优块大小、kernel 变体、OCC 参数
  - 收集运行时数据用于训练

---

## 长期（6个月以上）

- Phase 12 — 硬件特化（特定 Intel / NVIDIA / AMD 优化）
- Phase 13 — 自定义字典 / 两阶段压缩（更多压缩率）
- Phase 14 — GPU 性能建模与 Roofline 分析

---

## 现在要做的具体事项（操作项）

1. 完成 Phase 8.1 的多线程 I/O（实现并验证：io_uring / aio / pthread pipeline）。
2. 并行测试 Phase 10 的异步流水线样例代码（双缓冲事件依赖管理）。
3. 在真实 dGPU 上验证 Zero-Copy 与 Explicit Copy 的差异，决定默认行为（混合策略）。
4. 继续内核微调（低层优化、小型验证用例、性能计数器）以收敛内核时间。

---

## 参考

- 实现与性能对比：`PERFORMANCE_SUMMARY.md`
- 源代码：`lzo_host.c`, `daemon_compress.c`, `daemon_decompress.c`（lzo_gpu/）
"""
Explanation: replace OPTIMIZATION_ROADMAP.md with a clean roadmap reflecting final status and next steps.
"""
   ```c
   // 优化前: 独立work-item
   global_size = nblk;
   local_size = 1;

   // 优化后: workgroup协作
   local_size = 16;  // 16个work-item共享4KB Local Memory
   global_size = (nblk + 15) / 16 * 16;
   ```

**预期性能**:
```
lzo1x_1l:
  当前: 2416 MB/s
  预期: 2660 MB/s (+10%)

lzo1x_1k:
  当前: 2507 MB/s
  预期: 2880 MB/s (+15%) ← 更小字典，收益更大
```

**风险**:
- ⚠️ 字典更新竞争可能降低性能
- ⚠️ workgroup同步开销
- ✅ 可通过性能计数器验证Local Memory命中率

---

### Phase 6: 数据传输优化 ⭐⭐

**目标**: 减少PCIe传输开销 (当前11.4%)

**复杂度**: 🔥🔥 简单-中等
**预期收益**: **+5-8%**
**实施时间**: 1周

#### 6.1 Pinned Memory (页锁定内存)

**当前问题**:
```c
// 普通malloc分配的内存需要额外拷贝
uchar *in_buf = malloc(in_sz);          // 可分页内存
clEnqueueWriteBuffer(q, d_in, ..., in_buf, ...);
// 内部流程: in_buf → 驱动临时缓冲(pinned) → GPU
```

**优化方案**:
```c
// 使用CL_MEM_ALLOC_HOST_PTR创建零拷贝缓冲区
cl_mem d_in = clCreateBuffer(ctx,
    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
    in_sz, NULL, &err);

// 映射到主机地址空间 (零拷贝)
uchar *in_mapped = clEnqueueMapBuffer(q, d_in, CL_TRUE,
    CL_MAP_WRITE, 0, in_sz, 0, NULL, NULL, &err);

// 直接写入映射内存
fread(in_mapped, 1, in_sz, fp);  // 或 memcpy

// 解映射触发DMA传输
clEnqueueUnmapMemObject(q, d_in, in_mapped, 0, NULL, NULL);
```

**预期收益**:
- 上传时间: 176ms → 140ms (-20%)
- 下载时间: 47ms → 38ms (-20%)
- 总体提升: **+3-4%**

#### 6.2 异步传输流水线

**当前问题**:
```c
// 串行处理
for (i = 0; i < nblk; i++) {
    upload_block(i);      // 等待传输完成
    compress_block(i);    // 等待kernel完成
    download_block(i);    // 等待传输完成
}
// GPU利用率: ~60% (传输期间闲置)
```

**优化方案: 双缓冲流水线**:
```c
cl_mem d_in[2], d_out[2];  // 双缓冲
cl_event upload_event[2], kernel_event[2];

for (i = 0; i < nblk; i++) {
    int curr = i % 2;
    int next = (i + 1) % 2;

    // 并发: 上传块N+1，同时处理块N
    if (i < nblk - 1) {
        clEnqueueWriteBuffer(q, d_in[next], CL_FALSE,  // 异步上传
            0, blk_sz, host_in + (i+1)*blk_sz,
            0, NULL, &upload_event[next]);
    }

    // 等待当前块上传完成，启动kernel
    clEnqueueNDRangeKernel(q, krn, ...,
        1, &upload_event[curr], &kernel_event[curr]);

    // 等待kernel完成，异步下载
    clEnqueueReadBuffer(q, d_out[curr], CL_FALSE,
        0, out_sz, host_out + i*out_sz,
        1, &kernel_event[curr], NULL);
}
clFinish(q);  // 等待所有操作完成
```

**预期收益**:
- 传输与计算重叠度: 70-80%
- 传输开销隐藏: 223ms → 50ms
- 总体提升: **+8-10%**

**复杂度**: 高
- 需要事件依赖管理
- 缓冲区同步复杂
- 需要仔细测试边界条件

---

### Phase 7: 算法级优化 ⭐

**目标**: 改进压缩算法本身

**复杂度**: 🔥🔥🔥🔥 高
**预期收益**: **+5-10%** 速度 或 **+1-3%** 压缩率
**实施时间**: 2-4周

#### 7.1 Lazy Matching (延迟匹配)

**原理**:
```c
// 当前: 贪心匹配
if (m_len >= M2_MIN_LEN) {
    encode_match(m_off, m_len);  // 立即编码
}

// Lazy Matching: 检查下一个位置是否更好
if (m_len >= M2_MIN_LEN) {
    // 尝试下一个位置
    uint next_m_len = find_match(ip + 1);

    if (next_m_len > m_len + 1) {
        // 下一个匹配更长，放弃当前
        encode_literal(*ip);
        ip++;
        m_len = next_m_len;
    }
    encode_match(m_off, m_len);
}
```

**权衡**:
- ✅ 压缩率提升: **+1-3%**
- ❌ 速度下降: **-5-10%** (额外哈希查找)
- 🎯 适合"压缩优先"场景

#### 7.2 自适应块大小

**当前**: 固定块大小 163840 字节 (160KB)

**优化**: 根据数据熵动态调整

```c
// 快速熵估算 (采样前4KB)
float estimate_entropy(const uchar *data) {
    uint freq[256] = {0};
    for (int i = 0; i < 4096; i++) {
        freq[data[i]]++;
    }

    float entropy = 0;
    for (int i = 0; i < 256; i++) {
        if (freq[i]) {
            float p = freq[i] / 4096.0f;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

// 自适应块大小
uint adaptive_block_size(const uchar *data, size_t total_size) {
    float entropy = estimate_entropy(data);

    if (entropy > 7.5) {
        // 高熵数据 (随机/已压缩): 大块减少开销
        return 256 * 1024;
    } else if (entropy < 4.0) {
        // 低熵数据 (重复多): 小块增加并行度
        return 64 * 1024;
    } else {
        // 中等熵: 默认大小
        return 160 * 1024;
    }
}
```

**预期收益**:
- 高熵数据: **+10-15%** (减少块间开销)
- 低熵数据: **+5-8%** (更好的并行度)
- 平均提升: **+5-7%**
- 压缩率: **+0-2%**

#### 7.3 多级哈希表 (减少冲突)

**目标**: 降低哈希冲突率，提升匹配发现率

**当前**: 单级直接映射
```c
uint hash = DINDEX(dv);  // D_BITS位索引
dict[hash] = pos;        // 直接覆盖
```

**优化**: Two-way Set Associative Hash

```c
// 每个哈希索引有2个槽位
__local uint dict_set0[DICT_SIZE];
__local uint dict_set1[DICT_SIZE];

uint hash = DINDEX(dv);

// 查找: 检查两个槽
uint pos0 = dict_set0[hash];
uint pos1 = dict_set1[hash];

uint m_len0 = check_match(ip, pos0);
uint m_len1 = check_match(ip, pos1);

// 选择更长的匹配
if (m_len1 > m_len0) {
    m_pos = pos1; m_len = m_len1;
} else {
    m_pos = pos0; m_len = m_len0;
}

// 更新: LRU策略
if (m_len0 < m_len1) {
    dict_set0[hash] = current_pos;  // 替换较差的槽
} else {
    dict_set1[hash] = current_pos;
}
```

**预期收益**:
- 哈希冲突: -50-70%
- 压缩率: **+0.5-1.5%**
- 速度影响: **-2-5%** (多一次匹配检查)
- 内存需求: 2倍字典大小

---

### Phase 8: CPU端优化 ⭐⭐

**目标**: 优化CPU端处理 (当前占50.1%)

**复杂度**: 🔥🔥 中等
**预期收益**: **+10-20%** (整体吞吐)
**实施时间**: 1-2周

#### 8.1 多线程I/O

**当前问题**:
```c
// 单线程串行
read_file();      // CPU忙，GPU闲
compress_gpu();   // GPU忙，CPU闲
write_file();     // CPU忙，GPU闲
```

**优化方案**:
```c
#include <pthread.h>

typedef struct {
    uchar *data;
    size_t size;
    sem_t ready;
} buffer_t;

buffer_t read_buf, compress_buf, write_buf;

// 线程1: 读取
void* reader_thread(void*) {
    while (has_data) {
        read_next_chunk(&read_buf);
        sem_post(&read_buf.ready);
    }
}

// 线程2: GPU压缩
void* compress_thread(void*) {
    while (running) {
        sem_wait(&read_buf.ready);
        compress_gpu(&read_buf, &compress_buf);
        sem_post(&compress_buf.ready);
    }
}

// 线程3: 写入
void* writer_thread(void*) {
    while (running) {
        sem_wait(&compress_buf.ready);
        write_output(&compress_buf);
    }
}

// 主线程: 协调三个流水线stage
pthread_create(&tid_read, NULL, reader_thread, NULL);
pthread_create(&tid_compress, NULL, compress_thread, NULL);
pthread_create(&tid_write, NULL, writer_thread, NULL);
```

**预期收益**:
- I/O与GPU完全重叠
- CPU开销隐藏: 50.1% → 5%
- 总体提升: **+15-20%**

#### 8.2 零拷贝I/O (mmap)

**当前**:
```c
FILE *fp = fopen(path, "rb");
fread(buf, 1, size, fp);  // 用户空间拷贝
```

**优化**:
```c
int fd = open(path, O_RDONLY);
uchar *mapped = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
// 直接使用mapped，无需拷贝
clEnqueueWriteBuffer(q, d_in, ..., mapped, ...);
munmap(mapped, file_size);
```

**预期收益**:
- 消除一次内存拷贝
- 大文件提升显著: **+5-10%**

---

## 长期优化方向 (Phase 9+)

### Phase 9: 硬件特化优化

#### 9.1 Intel Xe Matrix Extensions (XMX)

**目标**: 使用矩阵加速单元

**应用场景**:
- 批量哈希计算 (dp4a指令)
- 向量化内存拷贝 (矩阵块传输)

**预期收益**: **+10-20%** (Intel GPU)

#### 9.2 Subgroup优化 (Wave-level)

**当前**: Work-item级并行
**优化**: Subgroup级SIMD

```c
// 使用subgroup洗牌加速字典查找
uint sg_size = get_sub_group_size();  // 通常8或16
uint lane_id = get_sub_group_local_id();

// 广播字典查询到整个subgroup
uint hash = sub_group_broadcast(hash_value, 0);
uint result = dict[hash + lane_id];  // 连续访问，coalesced
```

**预期收益**: **+5-15%**

---

### Phase 10: 自定义字典

**目标**: 预训练领域字典

**应用**:
- JSON数据: 预加载`"key":`, `"value"`, etc.
- 日志文件: 时间戳模式、关键字
- 二进制协议: 固定头部、Magic number

**实现**:
```c
// 加载预训练字典到GPU
cl_mem d_pretrained_dict = clCreateBuffer(ctx,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    dict_size, pretrained_data, &err);

// Kernel使用预训练字典作为初始状态
__kernel void compress_with_dict(
    __constant uint *pretrained_dict,
    __global const uchar *in, ...)
{
    __local uint dict[DICT_SIZE];

    // 初始化为预训练字典
    dict[lid] = pretrained_dict[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 压缩过程中动态更新
    ...
}
```

**预期收益**:
- 压缩率: **+5-20%** (领域数据)
- 速度: 可忽略影响

---

## 性能目标

### 中期目标 (Phase 5-8, 1-2个月)

**lzo1x_1l 性能预测**:

| 优化阶段 | 压缩(MB/s) | 解压(MB/s) | 压缩率 |
|---------|-----------|-----------|--------|
| **当前** (Phase 4) | 2416 | 6973 | 6.583 |
| +Phase 5 (Local Mem) | **2660** (+10%) | 6973 | 6.583 |
| +Phase 6.1 (Pinned Mem) | **2740** (+3%) | **7200** (+3%) | 6.583 |
| +Phase 6.2 (Pipeline) | **2877** (+5%) | **7560** (+5%) | 6.583 |
| +Phase 7.2 (Adaptive) | **3020** (+5%) | **7938** (+5%) | **6.65** (+1%) |
| +Phase 8.1 (Multi-thread) | **3625** (+20%) | **9526** (+20%) | 6.65 |
| **目标** | **~3.5 GB/s** | **~9.5 GB/s** | **~6.6** |

**累计提升**:
- 压缩: 2416 → 3625 MB/s (**+50%**)
- 解压: 6973 → 9526 MB/s (**+37%**)

### 长期目标 (Phase 9-10, 3-6个月)

- 压缩吞吐量: **4+ GB/s**
- 解压吞吐量: **10+ GB/s**
- 压缩率: **6.8+** (领域字典)

---

## 实施优先级

### ✅ 已完成 (Phase 1-8.2)

1. **✅ Phase 1-4: 内核端核心优化** (已完成)
   - Phase 1: 并行度优化 (+27.8%)
   - Phase 2: 地址空间统一 (0%)
   - Phase 3: 哈希函数优化 (+0.3-2.4%)
   - Phase 4: 向量化匹配长度/解压 (+22% / +218%)
   - 状态: **已完成并验证** ✅

2. **✅ Phase 6.1: Pinned Memory** (已完成)
   - 难度: 🔥 低
   - 实际收益: +1-2% (传输效率)
   - 风险: 极低
   - 状态: **已完成** ✅
   - 结论: Intel驱动已优化，收益小于预期但零成本

3. **✅ Phase 7.2: 自适应块大小** (已完成)
   - 难度: 🔥🔥 中
   - 实际收益: 与Phase 6.1持平 (2421 vs 2422 MB/s)
   - 状态: **已完成** ✅
   - 结论:
     - 成功实现GPU感知的自适应块大小选择（3个版本演进）
     - **v1版本(仅看熵)**: 2352 MB/s，块数4,445，GPU利用不足 ❌
     - **v2版本(GPU优先)**: 2421 MB/s，块数11,183，平衡并行度 ✅
     - **v2增强版(动态OCC)**: 2421 MB/s，根据熵调整块数量倾向 ✅ **推荐**
     - 核心改进:
       * 优先保证并行度: target_nblk = CU × OCC_FACTOR
       * 动态OCC: 低熵OCC=64 (更少块), 高熵OCC=256 (更多块), 中等OCC=128
       * 动态max_blk: 低熵256KB, 中等192KB, 高熵128KB
       * 智能平衡: GPU利用率 + 压缩率 + 吞吐量
     - 性能验证:
       * 全0数据 (熵0.00): OCC=64, 6144目标块, 3345 MB/s
       * 真实数据 (熵4.73): OCC=128, 12288目标块, 2421 MB/s
       * 随机数据 (熵8.00): OCC=256, 24576目标块, 2260 MB/s
     - 通用性强，适应未知数据类型，零人工调参
     - 与最优固定块大小(64KB)差距<3%，但通用性更好

4. **✅ Phase 8.2: Zero-Copy 优化** (已完成 2025-11-25)
   - 难度: 🔥🔥 中
   - 实际收益: **Data Download: 49ms → 0.028ms (-99.94%)**
   - 风险: 低
   - 状态: **已完成** ✅
   - 结论:
     - **Zero-Copy 已成功实现**: 消除显式 `clEnqueueReadBuffer`，使用 `clEnqueueMapBuffer` 映射指针
     - **技术方案**:
       * `CL_MEM_ALLOC_HOST_PTR` 创建 Pinned Memory (页锁定内存)
       * `clEnqueueMapBuffer` 零拷贝映射，无显式 DMA 传输
       * Scatter-Gather 写入：直接从映射内存写文件，消除中间缓冲区 (`out_buf`)
     - **性能对比** (iGPU):
       * Standalone (原始): Download 49.003ms
       * Standalone (Zero-Copy): Download **0.028ms** ← 改善 **99.94%** 🔥
       * Daemon (Zero-Copy): Download 0.008ms
       * 同一数量级，Zero-Copy 已生效 ✅

5. **✅ Phase 8.3: Input Zero-Copy (直接 fread 到 Pinned Memory)** (已完成 2025-11-25)
   - 难度: 🔥 低
   - 实际收益: **File Read: 184ms → 0.011ms (-99.99%)**
   - 风险: 极低
   - 状态: **已完成** ✅
   - 结论:
     - **关键发现**: Daemon 的优势不是 mmap，而是**直接 fread 到 Pinned Memory**
     - **技术方案**:
       * 提前创建 Pinned Memory 缓冲区
       * `clEnqueueMapBuffer` 映射到主机地址空间
       * `fread(mapped_in)` 直接读入 GPU 可访问内存（零拷贝）
       * `clEnqueueUnmapMemObject` 解映射（数据已就绪）
     - **性能对比** (300MB 真实文本):
       * Phase 8.2: Read 184ms, Download 0.028ms, Write 117ms → I/O Total 301ms
       * Phase 8.3: Read **0.011ms**, Download 0.027ms, Write 26ms → I/O Total **27ms**
       * 改善: **-92% I/O 时间**
     - **vs Daemon**:
       * Standalone Read: **0.011ms** vs Daemon Read: 38.402ms ← Standalone 快 3491 倍！
       * Standalone I/O Total: **26.468ms** vs Daemon I/O Total: 64.453ms ← Standalone 快 59%
     - **原因分析**: Standalone 无守护进程开销，直接文件操作更快

**当前性能**:
- lzo1x_1l: 2421 MB/s (压缩) + 6973 MB/s (解压)
- **I/O Total**: **26ms** (Standalone, Phase 8.3) vs 64ms (Daemon) vs 301ms (Phase 8.2)
- **Zero-Copy 完全实现**: Download ~0.01ms, Read ~0.01ms

---

### ⭐⭐⭐ 下一步实施 (本周)

6. **Phase 5: Local Memory 优化**
   - 难度: 🔥🔥🔥 中-高
   - 预期收益: +10-15% (kernel 执行)
   - 风险: 中等
   - 🔄 **推荐** (I/O 已优化至极限，转向 kernel 优化)

7. **Phase 6.2: 异步传输流水线**
   - 难度: 🔥🔥🔥🔥 高
   - 预期收益: +5-8% (重叠 I/O 与计算)
   - 风险: 高
   - ⚠️ **延后** (I/O 已接近 0，收益有限)

### ⭐⭐ 近期实施 (2-4周)

5. **Phase 5: Local Memory优化**
   - 难度: 🔥🔥🔥 中-高
   - 收益: +10-15%
   - 风险: 中等 (并发问题)
   - 🔄 需要仔细调试

### ⭐ 中期实施 (1-2月)

6. **Phase 6.2: 异步流水线**
   - 难度: 🔥🔥🔥🔥 高
   - 收益: +8-10%
   - 风险: 高 (复杂度)
   - ⚠️ 需要充分测试

7. **Phase 7.1/7.3: 算法优化**
   - 难度: 🔥🔥🔥🔥 高
   - 收益: +5-10% 或 压缩率+1-3%
   - 风险: 中
   - 🔬 需要实验验证

### 🔬 长期研究 (3-6月)

7. **Phase 9: 硬件特化**
8. **Phase 10: 自定义字典**

---

## 下一步行动

### 本周计划

- [x] ✅ 完成Phase 4向量化优化
- [x] ✅ 更新性能文档
- [x] ✅ 制定中期路线图

### 下周计划

- [ ] 🚀 实现Phase 6.1: Pinned Memory
  - 修改lzo_host.c内存分配
  - 性能对比测试
  - 预期用时: 2-3天

- [ ] 🚀 实现Phase 7.2: 自适应块大小
  - 添加熵估算函数
  - 动态块大小调整
  - 预期用时: 3-4天

**目标**: 2周内实现**+8-11%**整体性能提升

---

**最后更新**: 2025-11-23
**当前阶段**: Phase 4 完成，准备Phase 5-8
**项目状态**: 核心优化完成，进入深度优化阶段
