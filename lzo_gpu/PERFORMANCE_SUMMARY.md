# LZO GPU — 性能摘要（内核 & 主机侧优化）

更新：2025-12-01

这是仓库中最终的“简洁性能摘要”；只包含直接且可复现的结论与建议。详细 per-sample 分析与历史测试结果在 `lzo_gpu/exp_results` 下的各汇总目录中（例如 `summary_mt123_*`）。

---

## 快速结论（要点）

- Baseline（参考）: standalone + std — mean_ms_no_ocl ≈ 71.66 ms
- 最佳（延迟）: daemon + zero+mt — mean_ms_no_ocl ≈ 40.62 ms (~43% 优化)
- IO 改善（daemon 聚合）: zero: mean_io_ms ≈ 5.444 ms；zero+mt: ≈ 3.966 ms；std: ≈ 11.004 ms；std+mt: ≈ 8.161 ms

结论：Zero-Copy + Multi-threaded I/O（MT_IO）+ Daemon 是减小服务端感知延迟的最佳组合；在 dGPU（PCIe）环境下，标准拷贝 + 多线程（std+mt）在上传瓶颈情况下可带来改进。

---

## 内核（GPU）端优化（逐项：原理 / 实现 / 实验结果 / 结论）

### 向量化匹配（Vectorized matching）

- 原理：使用宽加载 (wide loads) + 位运算（XOR）与 CTZ(count trailing zeros) 跳过位相同的字节，避免逐字节循环，从而降低每次匹配的比较开销。

- 实现：OpenCL kernels 中通过 `lzo_gpu/lzo1x_1k.cl`, `lzo_gpu/lzo1x_1l_opt.cl` 的向量路径实现 (ctz(diff) >> 3 用于定位第一个不同字节)。

- 实验结果：代表性变体 `lzo1x_1k` 压缩吞吐约 2507 MB/s；`lzo1x_1l` 压缩 ~2421 MB/s（见 repo 的 microbench），表明向量化匹配对压缩吞吐具有显著贡献（kernel 稳定提高）。

- 结论：向量化匹配是内核性能的关键优化，应保持为默认实现（当硬件支持时启用）。

### 向量化解压（Vectorized decompression）

- 原理：批量加载（16/8/4 字节）并向量化 memcpy/拷贝路径，优化短距离/短拷贝模式，减少指令数与内存访问次数。

- 实现：`lzo_gpu/lzo1x_decomp.cl` / `lzo_gpu/daemon_decompress.c` 已集成向量化解压逻辑，默认启用。

- 实验结果：代表性 `lzo1x_1l` 解压吞吐约 6973 MB/s（microbench），显著高于 scalar 路径（典型提升 2~4x，受硬件/驱动影响）。

- 结论：向量化路径已作为默认实现，对解压吞吐效果最佳。

### 轻量哈希（Lightweight XOR Hash）

- 原理：使用简单的位运算(XOR + shifts)替换复杂乘法哈希以减少单次哈希计算周期与寄存器压力，从而降低哈希开销。

- 实现：在 kernel/host 字典索引实现层，使用 `DINDEX()` / `LZO_HASH` 配置以及在 `src/config*.h` 中选择哈希策略（已完成 Phase 3）。

- 实验结果：Roadmap/阶段测试表明整体吞吐提升在 +0.3% ~ +2.4% 区间（相对基准），并可显著降低哈希冲突（结合二路哈希表可进一步提升匹配发现率）。

- 结论：使用轻量哈希可以无损压缩率地降低哈希开销，是一个低风险高回报的优化配置。

### 自适应块大小（Adaptive block size）

- 原理：根据数据熵自适应选择块大小，平衡块大小（减小块数）与 GPU 并行度之间的权衡，从而在高熵数据下提高并行度、在低熵数据下保持良好压缩率。

- 实现：在主机端（`lzo_gpu/lzo_gpu_standalone.c`）实现熵采样与基于阈值调整的分块器；通过 OCC_FACTOR / 块大小的动态调整实现。

- 实验结果：在 1.8GB 真实文本数据上，64KB -> 2470 MB/s（压缩），80KB -> 2449 MB/s，相比 Phase 6.1 （2396 MB/s）提升约 +2.2%；512KB 时吞吐降至 2229 MB/s，压缩率基本持平（+1%）。

- 结论：默认 80~96KB 的 adaptive block 大小在大多数场景下能在吞吐与压缩率之间取得更好的平衡；应默认启用自适应块大小策略。

---

## 主机（Host）侧优化

- Zero-copy（CL_MEM_ALLOC_HOST_PTR + clEnqueueMapBuffer）：将数据 fread/fwrite 直接到 pinned host buffer，从而减少或消除 memcpy 开销。
- 多线程 I/O（MT_IO）：并发 pread 写入映射缓冲区，降低 read 部分延迟并提高吞吐。
- 异步上传（std+async）：重叠上传和主线程工作，适用于 PCIe 上传瓶颈情形。
-- 守护进程（Daemon）：复用 OpenCL context/queues/buffers，避免重复初始化开销，推荐用于服务化部署。

## 主机（Host）侧优化（逐项：原理 / 实现 / 实验结果 / 结论）


### Zero-copy（Pinned host memory）

- 原理：通过 CL_MEM_ALLOC_HOST_PTR 分配映射到 device 的 pinned host memory，再使用 `clEnqueueMapBuffer`/`fread` 直接把文件内容读入映射缓冲区，避免 host→device memcpy。
- 实现：主机端实现位于 `lzo_gpu/lzo_gpu_standalone.c` 与守护端 `lzo_gpu/daemon_compress.c`；关键环境变量/选项：`LZO_STANDARD_COPY=0`（默认为 zero-copy）与 `CL_MEM_ALLOC_HOST_PTR`。
-- 实验结果：在 daemon 模式的聚合数据（n=3078, runs=6），zero mean_io_ms ≈ 5.444 ms，相较于 std 的 mean_io_ms ≈ 11.004 ms，IO 时间减少约 ~50%。
- 结论：Zero-copy 在 iGPU/共享内存平台上收益显著：默认启用 zero-copy；在 PCIe 场景下需使用 std/async（见下）以规避可能的驱动 DMA 行为。

### 多线程 I/O（MT_IO）

- 原理：把文件读取分片并分配到多个 pread 线程并行读取入映射缓冲区，减小单个读取延迟并提升并行吞吐。
- 实现：主机端通过 `LZO_MT_IO=1` 与 `LZO_MT_IO_THREADS=<n>` 环境变量控制，相关实现位于 `lzo_gpu/lzo_gpu_standalone.c`，守护端同步兼容。
-- 实验结果：daemon 模式下聚合数据（n=3078, runs=6）显示，zero → zero+mt 的 mean_io_ms 从 5.444 ms 降至 3.966 ms（约 27% 改善）；同样，daemon std → std+mt 从 11.004 ms 降至 8.161 ms（约 26% 改善）。
- 结论：MT_IO 在 I/O 瓶颈场景/大文件场景提升明显，建议与 zero-copy 组合使用；线程数应通过 `LZO_MT_IO_THREADS` 在目标硬件上调优。

### 守护进程（Daemon）

- 原理：守护进程常驻并复用 OpenCL context、queue、kernels、buffers，从而避免每次请求产生的 OpenCL 初始化与缓冲区分配开销。
- 实现：位于 `lzo_gpu/daemon_compress.c` / `lzo_gpu/daemon_decompress.c`，提供 socket/IPC 接口，主机客户端通过 RPC/IPC 向守护进程发出请求。
-- 实验结果：聚合数据（n=3078, runs=6）显示 daemon zero+mt mean_ms_no_ocl ≈ 40.619 ms，而 standalone zero+mt ≈ 63.496 ms（相对改善约 36%）；daemon 在 std 情形也优于 standalone（daemon std ≈ 47.601 vs standalone std ≈ 71.656，约 33% 改善）。
- 结论：Daemon 在服务化/低延迟的场景中显著减少整体 latency 和变动性，推荐为生产部署模式（尤其在需要多次请求或高 QPS 场景）。

---

## 部署建议（速览）

- 最低延迟（推荐）：daemon + zero+mt
- 在 dGPU/PCIe 上传为瓶颈时：daemon + std+mt_async
- 简单的 iGPU 部署：daemon + zero

示例：

```bash
# 启动守护进程
./lzo_gpu_daemon &

# 客户端零拷贝 + 多线程 I/O 示例
LZO_MT_IO=1 LZO_MT_IO_THREADS=2 ./lzo_gpu_client /root/samples/<file>.img
```

---

## 复现与数据来源

- 自动化运行：`tools/run_full_experiments.py` (e.g., --runs 6 --threads 1,2,3 --samples /root/samples)
- 分析脚本：`tools/analysis.py`, `tools/generate_aggregate_stats.py` 生成 `lzo_gpu/exp_results/summary_mt123_*`
- 聚合脚本：`tools/update_performance_summary_md.py` 会把最新的聚合统计注入到本文件中的 AGG 区段（见下）。

<!-- AGG_SUMMARY_START -->

---

## 全量实验结果汇总

以下为从多线程 I/O (# threads = 1,2,3) 全量实验聚合得到的关键统计: 每个组合的平均 ms_no_ocl (去除 OCL 初始时间), 以及 IO (read + upload) 与读/写占比，和稳定性标记(0.25 CV 阈值)

### Top 10 combos by mean ms_no_ocl (lower is better)

| Rank | runner | mode | async | mt | mean_ms_no_ocl | mean_io_ms | read% | write% |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | daemon | zero+mt | 0 | 1 | 40.619 | 3.966 | 9.76% | 0.00% |
| 2 | daemon | zero | 0 | 0 | 42.094 | 5.444 | 12.93% | 0.00% |
| 3 | daemon | std+mt_async | 1 | 1 | 44.694 | 8.138 | 13.81% | 0.00% |
| 4 | daemon | std+mt | 0 | 1 | 44.734 | 8.161 | 13.84% | 0.00% |
| 5 | daemon | std_async | 1 | 0 | 47.593 | 11.018 | 18.59% | 0.00% |
| 6 | daemon | std | 0 | 0 | 47.601 | 11.004 | 18.56% | 0.00% |
| 7 | standalone | zero+mt | 0 | 1 | 63.496 | 11.032 | 17.37% | 0.00% |
| 8 | standalone | std+mt_async | 1 | 1 | 65.426 | 12.914 | 9.70% | 0.00% |
| 9 | standalone | std+mt | 0 | 1 | 67.354 | 9.696 | 9.44% | 0.00% |
| 10 | standalone | std_async | 1 | 0 | 67.899 | 15.312 | 13.24% | 0.00% |

### 不稳定的组合 (CV > 0.25 的读或写)

| runner | mode | async | mt | cv_read | cv_write |
|---|---|---|---|---:|---:|
| daemon | std | 0 | 0 | 2.8397633724238305 |  |
| daemon | std+mt | 0 | 1 | 3.052641758155662 |  |
| daemon | std+mt_async | 1 | 1 | 3.031045637158158 |  |
| daemon | std_async | 1 | 0 | 2.8403550323811753 |  |
| daemon | zero | 0 | 0 | 2.8286526392479976 |  |
| daemon | zero+mt | 0 | 1 | 3.0978475388157047 |  |
| standalone | std | 0 | 0 | 2.7504030095224836 |  |
| standalone | std+mt | 0 | 1 | 2.918840502176058 |  |
| standalone | std+mt_async | 1 | 1 | 2.9191826998384793 |  |
| standalone | std_async | 1 | 0 | 2.758667347901766 |  |
| standalone | zero | 0 | 0 | 3.6880478287887595 |  |
| standalone | zero+mt | 0 | 1 | 2.7987237241935206 |  |


---


如需查看详细 per-file 分解，参见 `lzo_gpu/exp_results` 下的 `*_analysis.md` 与 `*_breakdown_summary_all.json` 文件。


<!-- AGG_SUMMARY_END -->
