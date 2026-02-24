# LZO GPU 性能分析报告（设计实现与结果）

更新时间：2026-02-24

## 1) 当前实现设计

### 1.1 主机侧执行架构

- 采用 daemon 常驻模式：OpenCL context/queue/预分配缓冲区常驻，避免每次请求重复初始化。
- 多 worker 并发：每个 worker 持有独立 queue 与 kernel 句柄，降低 `clSetKernelArg` 竞争。
- 压缩路径支持两种执行模式：
  - 常规路径：单次提交；
  - pipeline 路径：双槽位 chunk 流水（读入/上传/内核/下载/写回重叠），由文件规模与熵门控决定是否启用。
- 字典池按 active work-items 动态分配，结合 `LZO_GPU_WI_PER_CU` 与设备显存上限做并行度约束。

### 1.2 压缩内核设计（`lzo1x.cl` / `lzo1y.cl`）

- one-work-item-per-block，work-item 网格步进处理多个 block。
- 哈希字典采用 packed 64-bit entry：`{epoch, dict_entry}`，避免每块显式清表，降低重置开销。
- 指纹过滤：高 12 bit 指纹 + 低 20 bit 偏移，先过滤再验证 match，减少无效访问。
- 匹配查找路径包含 8 路向量化哈希探测与 64-bit/32-bit 分级比较。
- literal/match 拷贝使用 `uchar16/uchar8/uchar4` 分段向量拷贝。

### 1.3 解压路径设计

- 按 block 解压，local size 自动约束；
- 支持 standard-copy 与 map/unmap 两种传输模式；
- 大文件输出缓冲采用阈值回收（`LZO_DECOMP_CACHE_MB`）。

## 2) 测试口径与数据

- 样本目录：`/root/samples`
- 聚合方式：`repeats=3`，`AggMethod=median_mad`
- 矩阵：
  - CPU：`threads=1`, `alg=lzo1x,lzo1y`, `block=64K,256K`
  - GPU：`alg=lzo1x,lzo1y`, `level=12,13,14`, `block=16K,64K,128K`, `local=1`
- 数据文件：
  - baseline：`/tmp/ab_compare/base_lzo.csv`
  - modified：`/tmp/ab_compare/mod_lzo.csv`
  - 汇总：`/tmp/ab_compare/ab_summary_submit_check.json`

## 3) 稳定性与提交门槛结果

### 3.1 稳定性

- GPU case：`1710`
- roundtrip：`1710/1710`（100%）

### 3.2 压缩率与 CPU 单核对比

按 `File + Alg` 取 CPU/GPU 最优配置对比（pairs=`190`）：

- `Ratio%(GPU/CPU)` 中位数：`0.9696`
  - 说明压缩率与 CPU 单核基本一致（GPU略优，未出现明显劣化）。

### 3.3 吞吐与 CPU 单核对比

- `CompKernel(GPU)/Comp(CPU)` 中位数：`1.9219x`
  - `>1.0` 的 pair 数：`136/190`
- `DecKernel(GPU)/Dec(CPU)` 中位数：`3.9905x`
  - `>1.0` 的 pair 数：`178/190`

结论：LZO 当前版本满足“稳定 + 压缩吞吐总体优于 CPU 单核 + 压缩率基本不变”的提交条件。

## 4) 瓶颈分析（压缩/解压）

基于 `mod_lzo.csv` 的 kernel 与端到端吞吐差距统计：

- `CompKernel/CompOverall` 中位数：`1.4128`
  - 对应主机侧开销占比中位：`29.2%`
- `DecKernel/DecOverall` 中位数：`3.3026`
  - 对应主机侧开销占比中位：`69.7%`

说明：

- 压缩端已具备较好的 GPU 利用；
- 解压端当前主要瓶颈在 host 侧传输/聚合与调度，不在 kernel 计算本身。

## 5) 后续优化与测试计划

### 5.1 压缩端（LZO）

1. **哈希字典**：
   - 评估 `D_BITS` 分档 + 字典池大小自适应策略；
   - 比较 epoch 方案与局部清表成本在不同块大小下的收益。
2. **匹配查找**：
   - 调整 fingerprint 位宽与 early-exit 条件；
   - 对高熵样本引入低成本短路分支，减少无效 compare。
3. **并行流水线**：
   - 扫描 `chunk_blocks × wi_per_cu × local_size`；
   - 目标是压缩端 overall 再提升、且尾部样本不退化。

### 5.2 解压端（LZO）

1. 优化 host 侧下载与写回路径（map/unmap 批次、输出聚合）；
2. 按块输出改为更强的异步重叠（download 与写盘并行）；
3. 扫描 `local_size` 与 block 粒度，优先压低 `DecKernel/DecOverall` 差距。

### 5.3 回归标准

- roundtrip：100%
- 压缩率：`Ratio%(GPU/CPU)` 中位保持在 `1.0 ± 5%`
- 压缩吞吐：`CompKernel(GPU)/Comp(CPU)` 中位 > `1.0`
- 解压吞吐：`DecKernel(GPU)/Dec(CPU)` 中位持续提升
