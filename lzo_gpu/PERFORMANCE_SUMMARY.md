# LZO GPU 性能总结（本轮实现与全量结果）

更新时间：2026-02-27

## 1. 范围与口径

- 对比区间：`mod3_lzo_decomp.csv` → `mod6_lzo_full.csv`
- 对齐方式：按 `File + Alg + Level + BlockSize + Threads_LSZ` 对齐 GPU 行
- 对齐样本：`1512` 组
- Roundtrip：`1512/1512` 通过（GPU 行失败 `0`）
- CPU 对照口径：采用 `mod3` 中同文件 CPU 最佳 `CompMBs/DecMBs` 作为参考（`mod6` 为 GPU-only 跑测）

## 2. 本轮设计与实现（不含 roundtrip 故障修正细节）

涉及文件：`lzo_gpu/lzo1x.cl`、`lzo_gpu/lzo1y.cl`、`lzo_gpu/lzo_gpu_core.c`、`lzo_gpu/lzo_gpu.c`、`lzo_gpu/lzo_gpu_daemon.c`

1. **压缩 match loop 展开（1x/1y 同步）**
   - `LZO_USE_UNROLL2` 从 `0` 调整为 `1`。
   - 在 match 扩展热路径减少循环控制开销，提高 ILP。

2. **调度并行度上调**
   - `LZO_GPU_WI_PER_CU` 默认值由 `256` 提升到 `384`（pipeline 与非 pipeline 路径一致）。
   - 目标是提高 CU 饱和度并改善吞吐稳定性。

3. **OpenCL 初始化鲁棒性增强**
   - 设备选择升级为 `GPU → DEFAULT → ALL` 回退链。
   - standalone/daemon 两条路径都补充失败检查，避免初始化即退出。

4. **既有向量化路径与本轮调度协同**
   - 解压中持续使用 `moff >= mlen` 非重叠快路 + `COPY_MATCH` 重叠安全路径。
   - 与本轮调度增强共同拉升整体 kernel 吞吐。

## 3. 全量结果（mod6 vs mod3）

### 3.1 聚合指标

- `CompKernelReported_MBs`：median **+249.38%**，p10 **+50.89%**
- `DecKernelReported_MBs`：median **+150.91%**，p10 **+16.91%**
- `Ratio%`：median **0.00**，p90 **0.00**，`>1% / >5% / >10% = 0 / 0 / 0`
- 解压提升覆盖率：`1435/1512 = 94.91%`

### 3.2 与 CPU 的中位对照（逐文件）

- `GPU CompKernel / CPU Comp`：**4.88x**（文件级中位）
- `GPU DecKernel / CPU Dec`：**8.97x**（文件级中位）

## 4. 提升较多/较少文件（逐文件中位）

> 说明：表中 `old/new` 分别对应 `mod3/mod6`；CPU 吞吐为 `mod3` 同文件 CPU 最佳值；压缩率变化均为 0。

### 4.1 提升较多（按解压 kernel 提升排序）

| 文件 | CompK old→new (MB/s) | DecK old→new (MB/s) | DecK变化 | CPU Comp / Dec (MB/s) | GPU/CPU(CompK, DecK) | Ratio 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transportation_parent_0_pages_img.tar` | 160.57 → 4306.48 | 319.86 → 6906.33 | **+2059.17%** | 1329.83 / 1336.87 | 3.24x / 5.17x | 0.00 |
| `sample_2mb_structured_2.txt` | 186.69 → 3873.42 | 466.16 → 5270.54 | **+1030.64%** | 1257.56 / 1918.58 | 3.08x / 2.75x | 0.00 |
| `ooffice` | 182.04 → 1929.06 | 478.98 → 4085.79 | **+753.03%** | 391.01 / 455.38 | 4.93x / 8.97x | 0.00 |

### 4.2 提升较少（含回退）

| 文件 | CompK old→new (MB/s) | DecK old→new (MB/s) | DecK变化 | CPU Comp / Dec (MB/s) | GPU/CPU(CompK, DecK) | Ratio 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sample_9mb_zero_3.txt` | 4345.73 → 4579.92 | 9249.99 → 3865.32 | **-58.21%** | 15192.57 / 1365.58 | 0.30x / 2.83x | 0.00 |
| `sample_6.80mb_zero_1.txt` | 3424.34 → 3827.07 | 7991.52 → 3355.92 | **-58.01%** | 14580.14 / 1395.73 | 0.26x / 2.40x | 0.00 |
| `sample_59.45mb_zero_3.txt` | 12971.72 → 12826.76 | 14129.45 → 10297.86 | **-27.12%** | 12783.36 / 1403.46 | 1.00x / 7.34x | 0.00 |

## 5. 结论

1. 本轮在压缩/解压 kernel 吞吐均实现高幅度中位提升，且解压 p10 也保持正向。
2. 压缩率稳定（统计口径下无回退样本）。
3. 尾部主要集中在“极高可压缩（全零类）”文件，后续需按数据分型做解压路径专项优化。

## 6. 下一步优化方向

1. **主机侧链路优化（优先）**：当前 kernel 提升已显著，下一步重点压缩 `upload/download/write` 开销，提升端到端吞吐与延迟。
2. **尾部场景优化**：对全零/短周期重复块做更轻量解压分流，优化 p10 与最差文件。
3. **参数自动调优**：按文件特征动态选择 `alg/level/block/local`，减少统一参数带来的尾部损失。
4. **指标体系扩展**：在 median 之外，固定追踪 p10/p5 与文件级负增益占比。
