# LZO GPU 性能总结（本轮优化、回退与逐文件对比）

更新时间：2026-02-26

## 1) 本轮提交链路与目标

- 压缩优化基线：`adf15e1`
- 压缩优化提交：`006dde8`（`lzo_gpu: reduce probe vector width in compression match search`）
- 解压优化提交：`0a2d593`（`lzo_gpu: add non-overlap fast path in decomp match copy`）

本轮目标：

1. roundtrip 持续 100%；
2. 压缩率不发生明显劣化；
3. 压缩/解压 kernel 吞吐持续高于 CPU 单核中位；
4. 抑制尾部回退并保留可回滚路径。

## 2) 设计实现（优化与回退）

### 2.1 压缩优化（已落地）

涉及文件：`lzo_gpu/lzo1x.cl`、`lzo_gpu/lzo1y.cl`

- 将压缩匹配查找中的 probe 向量宽度调小，减少高熵场景的无效访存与比较；
- 保留 epoch + packed 字典结构，继续降低重置成本；
- 在吞吐与命中率之间做保守平衡，避免“单场景极致优化”拖累全局中位。

### 2.2 解压优化（已落地）

涉及文件：`lzo_gpu/lzo1x_decomp.cl`、`lzo_gpu/lzo1y_decomp.cl`

- 新增 non-overlap match copy fast-path；
- overlap 情况继续走安全路径；
- 目标是降低常见路径的分支/复制开销，同时保证字节级一致性。

补充（本次下一阶段）：

- 在 `lzo1x_decomp.cl` / `lzo1y_decomp.cl` 的 `copy_match` 热路径显式增加“非重叠 (`moff >= mlen`) 直接 `UA_COPYN()`”分流；
- 仅在可能重叠时才走 `COPY_MATCH()`，降低常见非重叠 case 的分支压力。

### 2.3 回退/保守处理（已执行）

- 对不稳定或尾部退化明显的激进配置不作为默认；
- 保留稳定 fallback 路径（尤其在小块/随机数据上）；
- 采用“先稳后快”策略：先保证压缩率与正确性，再推进 kernel 吞吐。

## 3) 测试口径与数据资产

- 样本目录：`/root/samples`
- 聚合方式：`repeats=3`，`AggMethod=median_mad`
- 矩阵：
  - CPU：`threads=1`, `alg=lzo1x,lzo1y`, `block=64K,256K`
  - GPU：`alg=lzo1x,lzo1y`, `level=12,13,14`, `block=16K,64K,128K`, `local=1`
- 阶段汇总：
  - 压缩阶段：`/tmp/ab_compare/ab_summary_compress_opt.json`
  - 解压阶段：`/tmp/ab_compare/ab_summary_decomp_opt.json`
  - 下一阶段：`/tmp/ab_compare/ab_summary_next_stage.json`
- 逐文件详细表（已入库）：
  - `exp_results/lzo_per_file_cpu_gpu_compare_detailed.csv`
  - `exp_results/lzo_per_file_cpu_gpu_compare_detailed.md`
  - `exp_results/lzo_per_file_rankings.md`
  - 本轮全量结果：`/tmp/ab_compare/mod3_lzo_decomp.csv`

## 4) 全量 A/B 结果（含压缩阶段与解压阶段）

### 4.1 稳定性

- `rows_base/mod/common = 2090/2090/2090`
- `gpu_rows = 1710`
- roundtrip：`1710/1710`（100%）

### 4.2 压缩优化阶段（`adf15e1 -> 006dde8`）

- `CompKernelReported_MBs`：median **+1.35%**，p10 **-3.85%**
- `DecKernelReported_MBs`：median **+0.23%**，p10 **-1.94%**
- `Ratio%` 回退：
  - median `+0.458%`
  - p90 `+1.246%`
  - `>1% / >5% / >10%`：`267 / 0 / 0`
- CPU 单核对比（pair=190）：
  - `Ratio(GPU/CPU)` median：`0.9740`
  - `CompKernel(GPU)/Comp(CPU)` median：`1.9844x`
  - `DecKernel(GPU)/Dec(CPU)` median：`3.9688x`

结论：压缩阶段显著维持了 GPU 相对 CPU 的吞吐优势，但压缩率出现轻微可见波动（仍在可控区间）。

### 4.3 解压优化阶段（`006dde8 -> 0a2d593`）

- `CompKernelReported_MBs`：median **+0.013%**，p10 **-0.738%**
- `DecKernelReported_MBs`：median **+4.144%**，p10 **-0.901%**
- `Ratio%` 回退：
  - median `0.00%`
  - p90 `0.00%`
  - `>1% / >5% / >10%`：`0 / 0 / 0`
- CPU 单核对比（pair=190）：
  - `Ratio(GPU/CPU)` median：`0.9740`
  - `CompKernel(GPU)/Comp(CPU)` median：`1.9841x`
  - `DecKernel(GPU)/Dec(CPU)` median：`4.1911x`

结论：解压 fast-path 在不引入压缩率风险的前提下，显著提升了解压 kernel 中位吞吐。

### 4.4 下一阶段优化验证（当前工作树）

对比基线：`mod2_lzo_decomp.csv -> mod3_lzo_decomp.csv`（按 common rows 对齐）

- `rows(base/mod/common) = 2090/1848/1848`
- `gpu_rows = 1512`，roundtrip：`1512/1512`（100%）
- `CompKernelReported_MBs`：median **-0.114%**，p10 **-1.030%**
- `DecKernelReported_MBs`：median **-0.057%**，p10 **-1.244%**
- `Ratio%` 回退：`>1% / >5% / >10% = 0 / 0 / 0`
- CPU 单核对比（pair=168）：
  - `Ratio(GPU/CPU)` median：`0.9744`
  - `CompKernel(GPU)/Comp(CPU)` median：`1.9997x`
  - `DecKernel(GPU)/Dec(CPU)` median：`4.2340x`

结论：该阶段在压缩率与正确性上保持稳定，解压中位变化约 `-0.057%`（接近噪声区间）；以“稳定优先”标准判定可接受，后续继续针对 tail case 深挖。

## 5) 逐文件详细展示（压缩率/压缩核吞吐/解压核吞吐 对 CPU）

完整逐文件明细已写入：`exp_results/lzo_per_file_cpu_gpu_compare_detailed.md`。

该表逐行给出（95 文件 × 2 算法 = 190 行）：

- `CPU最佳Ratio%` 与 `GPU最佳Ratio%`；
- `CPU最佳Comp MB/s` 与 `GPU最佳CompK MB/s`；
- `CPU最佳Dec MB/s` 与 `GPU最佳DecK MB/s`；
- `GPU/CPU` 三个比值（ratio/comp/dec）；
- `GPU *ΔvsBase%`（相对压缩优化基线）。

补充排名（见 `exp_results/lzo_per_file_rankings.md`）：

- 解压核吞吐增益 Top：
  - `sample_6.80mb_zero_1.txt/lzo1x (+17.173%)`
  - `redis-memtier__migrate__parent_5__pages-1.img/lzo1y (+15.213%)`
- 解压核吞吐增益尾部：
  - `sample_61mb_repeat_4.txt/lzo1y (-4.797%)`
  - `sample_42mb_repeat_3.txt/lzo1x (-4.741%)`
- GPU/CPU 解压核吞吐比 Top：`11.445x`（`sample_132mb_zero_5.txt/lzo1y`）
- GPU/CPU 解压核吞吐比尾部：`0.372x`（`nginx-nc__migrate__parent_3__pages.img/lzo1y`）

## 6) 下一轮优化点（LZO）

1. **分场景 probe 自适应**：按块熵与命中统计动态切换 probe 宽度；
2. **解压尾部治理**：对 repeat/随机小块分别做特化，优先提升 p10；
3. **host 侧链路压缩**：下载/写回进一步异步化，降低解压端 non-kernel 占比；
4. **参数自动调优**：为 `level × block × local` 建立离线 profile + 在线选择策略。

## 7) 当前结论

- roundtrip 仍为 100%；
- 压缩率整体稳定（解压阶段统计下无 `>1%` 回退）；
- GPU 相对 CPU 单核：压缩中位约 `1.98x`、解压中位约 `4.19x`；
- 下一阶段验证显示：解压中位接近持平（`-0.057%`），未引入压缩率回退，整体可接受；
- 逐文件明细与 Top/Worst 已完整沉淀到 `exp_results/`，可直接复核与追踪后续迭代。
