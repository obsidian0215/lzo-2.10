# LZO GPU 性能总结（Intel + Nvidia）

> 更新时间：2026-04-30  
> Intel 当前口径：`lzo_gpu` 已按 v2 采纳项更新；`lzo_hybrid` 使用 v2 OpenCL CPU/GPU mixed 源码；`lzo_cpu` native 实现按 `t=1/2` 纳入对比。  
> 最新全集真实路径数据来源：`lzo_gpu_v2/variant_validation/arc/results/cross_platform_real_20260430/`

---

## Intel 平台（2026-04-30 / 当前主线口径）

### 1. 测试范围与口径

本节替换旧 Intel 平台结果。旧的 formal baseline、频率 sweep、候选 gate、错误 CPU slots 结果不再作为当前性能结论。

当前对比对象：

- `lzo_gpu`：当前 GPU 主线，使用默认 GPU 调度，不限制 GPU slots。
- `lzo_opencl_cpu_t1/t2`：`lzo_gpu_v2` OpenCL CPU-only 路径，`--gpu-ratio 0 --cpu-threads 1/2`。
- `lzo_cpu_native_t1/t2`：`lzo_cpu` native pthread 实现，`-t 1/2`，默认压缩等级。
- `lzop_1/3/5`：`lzop 1.04`，压缩等级 `-1/-3/-5`。

测试口径：

- 样本：全样本 25 个。
- OpenCL 配置：`48KB/64KB` block，`D_BITS=13/14/15`。
- native `lzo_cpu`：`48KB/64KB` block，默认等级，对比 `t=1/2`。
- 轮次：每个文件/配置 6 轮真实压缩 + 真实解压。
- `kernel` 吞吐：OpenCL profiling 或 native 内部压缩/解压阶段吞吐。
- `no-ocl` 总吞吐：真实文件路径中排除 OpenCL init/build 后的总耗时，包含文件读写、buffer、upload/download、kernel、打包/回收等。
- `lzop` 没有 kernel 指标，其 `no-ocl` 按命令真实运行时间计算。
- 临时压缩/解压产物写入结果目录并在轮次结束后删除，不污染样本目录。

### 2. 当前 `lzo_gpu` 已采纳的有效改动

#### 2.1 压缩主路径保持融合式 block kernel

压缩侧没有采纳 LZOG2 多阶段拆分，也没有把 host-only 的 bench/gather 优化当成主线能力。当前有效结构仍是原始 `lzo_gpu` 的融合式 block 压缩：

- 每个 active work-item 绑定一个私有字典 slice；
- 字典条目为 `epoch_12 | offset_20` 的 32-bit packed entry；
- block 间通过 epoch 逻辑复用字典，不在 kernel 内逐块全量清零；
- 热路径为单 primary hash + 4-probe batch read/writeback；
- match extension 保留已有宽比较展开。

这个选择的依据是多轮变体测试显示：拆分压缩如果不能减少 probe 随机读、候选验证和 match extension 的热路径成本，就只会增加中间状态写回和 kernel 启动/调度成本，无法稳定提升真实吞吐。

#### 2.2 解压 token fast path

动机：解压诊断显示 token/state-machine 密度与解压 kernel 吞吐强相关。旧路径在 match 完成后统一回读 `ip[-2] & 3` 获取 post-match literal 长度，会在高频 match 路径上制造额外输入回读。

设计与实现：

- 在 M2/M3/M4 常见 match 分支解析 token/offset 时提前保存 `post_lit`；
- `match_done` 直接使用 `post_lit`，不再统一回读 `ip[-2]`；
- 不改变 LZO block 格式、literal/match 语义、EOF 处理和 overlap copy。

效果：全样本验证显示该项稳定提升解压 kernel，压缩率变化 `0pp`，verify 全部通过；当前已同步到 `lzo1x.cl` 和 `lzo1y.cl`。

#### 2.3 short8 direct match copy

动机：解压分支诊断显示 direct copy 是主路径，且 `<=8B` match 占比高。旧 `<=18B` direct helper 对所有短 match 固定执行 16B 读写，`<=8B` 存在过度读写。

设计与实现：

- 保留 `offset >= mlen && mlen <= 18` 的 direct-copy 判定；
- `mlen <= 8` 时只做一次 `vload8/vstore8`；
- `9..18` 继续走原 16B copy，并保留 17/18B 尾字节处理；
- 不采纳 `<=4B` 进一步分层，因为新增高频分支成本会抵消收益。

效果：全样本确认 `short8_direct_copy` 对解压 kernel 有稳定收益，压缩率变化 `0pp`，verify 全部通过；当前已同步到 `lzo1x.cl` 和 `lzo1y.cl`。

#### 2.4 解压真实输出路径优化

已保留两项 host/readback 侧有效修改：

- 默认不写无用 `out_lens`：普通真实解压路径不消费每个 block 的输出长度，默认不给 kernel 传 `out_lens` buffer，避免无价值 global store；需要诊断时可通过 `LZO_GPU_DECOMP_TRACK_OUT_LENS=1` 恢复。
- chunked readback/write：真实解压到文件时可按 chunk 从 OpenCL buffer 回读并写文件，降低大输出文件一次性 download/map + fwrite 的尾部开销。当前策略是 Windows 下对大输出（默认阈值 16MB）可自动启用，Linux 默认关闭，避免在 Linux host path 上引入退化。

这两项只影响真实文件路径，不应通过 bench kernel 指标判断采纳价值。

### 3. 全样本总体结果

表中数值是“每个文件/配置 6 轮中位数”再按文件取中位，单位为 `MB/s`。

#### 3.1 Windows / Intel Arc B390

| 引擎 | block | D_BITS/等级 | 压缩率中位 | 压缩 kernel | 压缩 no-ocl | 解压 kernel | 解压 no-ocl |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lzo_gpu` | `48KB` | `13` | `33.03%` | `715.81` | `400.10` | `2899.84` | `898.19` |
| `lzo_gpu` | `48KB` | `14` | `32.90%` | `726.90` | `369.30` | `2778.32` | `902.11` |
| `lzo_gpu` | `48KB` | `15` | `32.84%` | `696.10` | `379.84` | `2893.93` | `895.42` |
| `lzo_gpu` | `64KB` | `13` | `32.75%` | `631.75` | `368.73` | `2508.51` | `841.03` |
| `lzo_gpu` | `64KB` | `14` | `32.60%` | `613.60` | `364.71` | `2368.64` | `833.09` |
| `lzo_gpu` | `64KB` | `15` | `32.53%` | `706.22` | `381.53` | `2282.29` | `829.55` |
| `lzo_opencl_cpu_t1` | `48KB` | `13` | `33.03%` | `457.05` | `349.16` | `1034.37` | `558.10` |
| `lzo_opencl_cpu_t1` | `48KB` | `14` | `32.90%` | `482.56` | `355.46` | `1041.40` | `563.14` |
| `lzo_opencl_cpu_t1` | `48KB` | `15` | `32.84%` | `490.63` | `360.93` | `1031.04` | `578.88` |
| `lzo_opencl_cpu_t1` | `64KB` | `13` | `32.75%` | `468.20` | `350.60` | `999.48` | `555.30` |
| `lzo_opencl_cpu_t1` | `64KB` | `14` | `32.60%` | `468.26` | `349.17` | `1030.40` | `548.42` |
| `lzo_opencl_cpu_t1` | `64KB` | `15` | `32.53%` | `468.35` | `343.24` | `1049.91` | `571.96` |
| `lzo_opencl_cpu_t2` | `48KB` | `13` | `33.03%` | `831.86` | `534.44` | `1487.22` | `661.30` |
| `lzo_opencl_cpu_t2` | `48KB` | `14` | `32.90%` | `896.51` | `529.03` | `1480.82` | `666.20` |
| `lzo_opencl_cpu_t2` | `48KB` | `15` | `32.84%` | `870.27` | `547.23` | `1490.91` | `675.34` |
| `lzo_opencl_cpu_t2` | `64KB` | `13` | `32.75%` | `841.59` | `530.41` | `1638.73` | `685.21` |
| `lzo_opencl_cpu_t2` | `64KB` | `14` | `32.60%` | `874.50` | `542.31` | `1606.06` | `683.12` |
| `lzo_opencl_cpu_t2` | `64KB` | `15` | `32.53%` | `856.02` | `538.56` | `1603.84` | `697.04` |
| `lzo_cpu_native_t1` | `48KB` | `14` | `32.53%` | `672.30` | `486.20` | `850.75` | `564.33` |
| `lzo_cpu_native_t1` | `64KB` | `14` | `32.89%` | `672.11` | `487.10` | `853.27` | `553.14` |
| `lzo_cpu_native_t2` | `48KB` | `14` | `32.53%` | `1300.20` | `735.28` | `1444.68` | `715.62` |
| `lzo_cpu_native_t2` | `64KB` | `14` | `32.89%` | `1291.89` | `755.02` | `1458.73` | `699.38` |
| `lzop_1` | `-` | `-1` | `32.69%` | `-` | `420.73` | `-` | `469.40` |
| `lzop_3` | `-` | `-3` | `32.58%` | `-` | `396.90` | `-` | `455.77` |
| `lzop_5` | `-` | `-5` | `32.58%` | `-` | `391.85` | `-` | `454.08` |

#### 3.2 Linux / Intel Iris Xe

| 引擎 | block | D_BITS/等级 | 压缩率中位 | 压缩 kernel | 压缩 no-ocl | 解压 kernel | 解压 no-ocl |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lzo_gpu` | `48KB` | `13` | `33.03%` | `1205.98` | `304.81` | `3034.28` | `412.72` |
| `lzo_gpu` | `48KB` | `14` | `32.90%` | `1189.49` | `285.65` | `3053.98` | `413.57` |
| `lzo_gpu` | `48KB` | `15` | `32.84%` | `1192.63` | `247.60` | `3052.95` | `412.56` |
| `lzo_gpu` | `64KB` | `13` | `32.75%` | `1012.53` | `303.14` | `2792.40` | `411.90` |
| `lzo_gpu` | `64KB` | `14` | `32.60%` | `970.16` | `284.94` | `2758.11` | `410.56` |
| `lzo_gpu` | `64KB` | `15` | `32.53%` | `966.02` | `250.72` | `2749.08` | `403.80` |
| `lzo_opencl_cpu_t1` | `48KB` | `13` | `33.03%` | `118.01` | `90.65` | `246.31` | `155.76` |
| `lzo_opencl_cpu_t1` | `48KB` | `14` | `32.90%` | `120.53` | `90.63` | `245.57` | `159.79` |
| `lzo_opencl_cpu_t1` | `48KB` | `15` | `32.84%` | `120.78` | `91.44` | `232.20` | `151.05` |
| `lzo_opencl_cpu_t1` | `64KB` | `13` | `32.75%` | `118.27` | `90.11` | `235.85` | `157.78` |
| `lzo_opencl_cpu_t1` | `64KB` | `14` | `32.60%` | `111.93` | `89.34` | `229.34` | `153.18` |
| `lzo_opencl_cpu_t1` | `64KB` | `15` | `32.53%` | `113.84` | `90.02` | `232.14` | `152.37` |
| `lzo_opencl_cpu_t2` | `48KB` | `13` | `33.03%` | `208.05` | `141.48` | `422.72` | `216.95` |
| `lzo_opencl_cpu_t2` | `48KB` | `14` | `32.90%` | `213.03` | `145.83` | `438.47` | `219.54` |
| `lzo_opencl_cpu_t2` | `48KB` | `15` | `32.84%` | `214.89` | `149.33` | `448.40` | `215.77` |
| `lzo_opencl_cpu_t2` | `64KB` | `13` | `32.75%` | `203.34` | `140.29` | `437.56` | `218.04` |
| `lzo_opencl_cpu_t2` | `64KB` | `14` | `32.60%` | `208.62` | `144.18` | `404.98` | `209.16` |
| `lzo_opencl_cpu_t2` | `64KB` | `15` | `32.53%` | `209.89` | `146.63` | `441.26` | `213.61` |
| `lzo_cpu_native_t1` | `48KB` | `14` | `32.53%` | `175.20` | `142.05` | `228.50` | `171.27` |
| `lzo_cpu_native_t1` | `64KB` | `14` | `32.89%` | `175.77` | `141.16` | `231.46` | `171.66` |
| `lzo_cpu_native_t2` | `48KB` | `14` | `32.53%` | `332.99` | `224.46` | `400.15` | `243.47` |
| `lzo_cpu_native_t2` | `64KB` | `14` | `32.89%` | `333.81` | `227.09` | `395.56` | `243.74` |
| `lzop_1` | `-` | `-1` | `32.69%` | `-` | `149.11` | `-` | `158.77` |
| `lzop_3` | `-` | `-3` | `32.58%` | `-` | `147.85` | `-` | `160.92` |
| `lzop_5` | `-` | `-5` | `32.58%` | `-` | `147.73` | `-` | `161.00` |

### 4. Intel 结果分析

1. `lzo_gpu` 保持当前 GPU 主线能力：Linux/Xe 上压缩 kernel 可达 `1.19~1.21 GB/s`，解压 kernel 可达 `2.75~3.05 GB/s`；Windows/Arc 上解压 no-ocl 为 `829.55~902.11 MB/s`，明显高于 `lzop` 和 native CPU。
2. native `lzo_cpu_native_t1/t2` 已纳入正式对比。Windows 上 `t=2` 压缩 no-ocl 为 `735.28~755.02 MB/s`，Linux 上 `t=2` 为 `224.46~227.09 MB/s`，均高于 `lzop -1/-3/-5`。
3. `lzo_hybrid` 纯 CPU 模式已用同一 OpenCL CPU 路径单独核对过，结果与 `lzo_opencl_cpu_t1/t2` 接近，因此汇总表不再重复列出；OpenCL CPU 仍明显慢于 native CPU，不能把它当成 native CPU 替代。
4. Windows 下 GPU 解压强、native CPU 压缩强；Linux 下 GPU 压缩/解压 kernel 明显强于 CPU。后续 hybrid 自适应调度应按“压缩/解压、平台、线程数、文件大小、GPU ratio”分别建模，而不是固定 CPU/GPU 二选一。
5. 压缩率方面，OpenCL GPU/CPU 在相同 block 与 `D_BITS` 下保持一致；`lzop -3/-5` 压缩率略好于 `lzop -1`，但性能差异很小。

### 5. 后续方向

- `lzo_gpu`：当前可提交；后续 GPU 压缩优化必须有明确 probe/extension 热路径证据，不能再采纳 bench-only 或真实路径无收益的修改。
- `lzo_hybrid`：当前源码结构可提交；真正的 mixed 自适应调度还需要单独建模与验证，当前只确认 GPU-only、CPU-only 和固定比例接口可用。
- `lzo_cpu`：当前 `t=1/2` 与 `48KB/64KB` 测试已建立 native CPU 对比基线；它是 hybrid 调度建模的 native CPU 参照，不是 OpenCL mixed 路径的一部分。

### 6. Linux / Intel 225 频率相关扫描（2026-05-01）

数据位置：

- 远端：`root@192.168.2.225:/root/lzo-2.10/exp_results/power_freq_runs/remote225_lzo_relevant_20260501/`
- 本地副本：`exp_results/power_freq_runs/remote225_lzo_relevant_20260501/`
- 汇总：`exp_results/power_freq_runs/remote225_lzo_relevant_20260501/REMOTE225_RELEVANT_SCAN_SUMMARY.md`
- 全量 CSV：`exp_results/power_freq_runs/remote225_lzo_relevant_20260501/combined_aggregate_summary.csv`

测试口径：

- 样本：`/root/samples` 全样本 25 个。
- 轮次：默认 `bench_seconds=5` + `manual_rounds=6`。
- 正确频率策略：GPU-only 只扫 GPU 频率；native CPU 与 hybrid `gpu_ratio=0` 只扫 CPU 频率；hybrid `gpu_ratio=0.5` 才扫 CPU×GPU 矩阵。
- CPU 频率点：`2100MHz / 3400MHz / NA(reset)`。
- GPU 频率点：`1000MHz / NA(reset)`。
- 所有子任务 `verify_all=True`，wrapper 子任务 `returncode=0`。

#### 6.1 正式相关频率矩阵

表中数值为全样本文件中位数，单位 `MB/s`；`CPU/GPU avg` 是 wrapper 采集到的平均频率，单位 `MHz`。

| workload | CPU target | GPU target | engine | block | ratio | CPU avg | GPU avg | 压缩 kernel | 解压 kernel | 压缩 no-ocl | 解压 no-ocl | 压缩率 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU-only | NA | 1000 | `lzo_gpu` | 48KB | 1 | 1580 | 1331 | 1132.6 | 2923.1 | 514.9 | 969.8 | 32.84% |
| GPU-only | NA | 1000 | `lzo_gpu` | 64KB | 1 | 1580 | 1331 | 939.9 | 2595.1 | 484.2 | 914.9 | 32.53% |
| GPU-only | NA | NA | `lzo_gpu` | 48KB | 1 | 1564 | 1395 | 1059.9 | 2947.3 | 502.8 | 950.3 | 32.84% |
| GPU-only | NA | NA | `lzo_gpu` | 64KB | 1 | 1564 | 1395 | 900.8 | 2520.1 | 471.4 | 921.1 | 32.53% |
| native CPU | 2100 | NA | `lzo_cpu_native` | 64KB | - | 2099 | 1239 | 375.3 | 492.4 | 280.4 | 330.9 | 32.89% |
| native CPU | 3400 | NA | `lzo_cpu_native` | 64KB | - | 3352 | 1221 | 608.8 | 799.8 | 445.1 | 515.6 | 32.89% |
| native CPU | NA | NA | `lzo_cpu_native` | 64KB | - | 1614 | 1404 | 665.6 | 880.6 | 470.3 | 556.9 | 32.89% |
| hybrid CPU-only | 2100 | NA | `lzo_hybrid` | 48KB | 0 | 2099 | 1500 | 275.0 | 645.5 | 195.7 | 337.2 | 32.90% |
| hybrid CPU-only | 2100 | NA | `lzo_hybrid` | 64KB | 0 | 2099 | 1500 | 268.1 | 612.2 | 189.0 | 336.4 | 32.60% |
| hybrid CPU-only | 3400 | NA | `lzo_hybrid` | 48KB | 0 | 3366 | 0 | 441.6 | 910.2 | 280.6 | 485.0 | 32.90% |
| hybrid CPU-only | 3400 | NA | `lzo_hybrid` | 64KB | 0 | 3366 | 0 | 427.9 | 1036.8 | 271.1 | 477.8 | 32.60% |
| hybrid CPU-only | NA | NA | `lzo_hybrid` | 48KB | 0 | 1450 | 1500 | 437.6 | 988.8 | 273.2 | 478.0 | 32.90% |
| hybrid CPU-only | NA | NA | `lzo_hybrid` | 64KB | 0 | 1450 | 1500 | 414.0 | 991.3 | 264.9 | 467.0 | 32.60% |
| hybrid mixed | 2100 | 1000 | `lzo_hybrid` | 48KB | 0.5 | 2099 | 1055 | 501.7 | 1262.5 | 267.0 | 438.3 | 32.87% |
| hybrid mixed | 2100 | 1000 | `lzo_hybrid` | 64KB | 0.5 | 2099 | 1055 | 497.2 | 1181.2 | 273.6 | 435.3 | 32.56% |
| hybrid mixed | 2100 | NA | `lzo_hybrid` | 48KB | 0.5 | 2099 | 974 | 502.1 | 1186.6 | 267.4 | 440.8 | 32.87% |
| hybrid mixed | 2100 | NA | `lzo_hybrid` | 64KB | 0.5 | 2099 | 974 | 496.6 | 1183.9 | 269.7 | 443.5 | 32.56% |
| hybrid mixed | 3400 | 1000 | `lzo_hybrid` | 48KB | 0.5 | 3360 | 1067 | 688.8 | 1615.2 | 375.8 | 616.3 | 32.87% |
| hybrid mixed | 3400 | 1000 | `lzo_hybrid` | 64KB | 0.5 | 3360 | 1067 | 672.3 | 1543.6 | 361.9 | 608.7 | 32.56% |
| hybrid mixed | 3400 | NA | `lzo_hybrid` | 48KB | 0.5 | 3362 | 1049 | 689.8 | 1580.1 | 357.6 | 606.5 | 32.87% |
| hybrid mixed | 3400 | NA | `lzo_hybrid` | 64KB | 0.5 | 3362 | 1049 | 669.8 | 1539.1 | 360.0 | 605.5 | 32.56% |
| hybrid mixed | NA | 1000 | `lzo_hybrid` | 48KB | 0.5 | 1410 | 1386 | 677.9 | 1622.3 | 365.0 | 623.2 | 32.87% |
| hybrid mixed | NA | 1000 | `lzo_hybrid` | 64KB | 0.5 | 1410 | 1386 | 667.8 | 1513.6 | 363.6 | 615.5 | 32.56% |
| hybrid mixed | NA | NA | `lzo_hybrid` | 48KB | 0.5 | 1420 | 1045 | 678.9 | 1538.6 | 336.0 | 605.0 | 32.87% |
| hybrid mixed | NA | NA | `lzo_hybrid` | 64KB | 0.5 | 1420 | 1045 | 658.5 | 1503.6 | 351.3 | 605.4 | 32.56% |

#### 6.2 频率扫描结论

1. 频率扫描必须按 workload 过滤。早期全笛卡尔矩阵会把 GPU-only 放进 CPU 频率点、把 CPU-only 放进 GPU 频率点，正式结论中已作废。
2. GPU-only 在 225 上受 GPU 频率点影响不大，且 telemetry 显示请求 `1000MHz` 时实际平均仍约 `1331MHz`，`NA` 时约 `1395MHz`。这说明 Intel GT 频率控制不是严格硬锁，解释结果时必须看 `gpu_freq_avg_mhz`，不能只看请求值。
3. native CPU 对 CPU 频率敏感：`2100MHz` 下压缩 no-ocl `280.4 MB/s`，`3400MHz` 下 `445.1 MB/s`，`NA` 下 `470.3 MB/s`。`NA` 实际平均频率只有 `1614MHz`，但性能最高，说明短时 boost/调度/功耗状态不能只用平均频率解释。
4. hybrid `gpu_ratio=0` 是 OpenCL CPU-only，性能显著低于 native CPU：`3400MHz/64KB` 压缩 no-ocl `271.1 MB/s`，native CPU `3400MHz/64KB` 为 `445.1 MB/s`。这再次确认 OpenCL CPU 不是 native CPU 的替代，只能作为同一 OpenCL mixed 框架内的 CPU 设备路径。
5. hybrid `gpu_ratio=0.5` 在 225 上比纯 GPU 压缩 no-ocl 慢：例如 `48KB, NA/NA` 为 `336.0 MB/s`，而 GPU-only `48KB, NA` 为 `502.8 MB/s`。但 mixed 解压 no-ocl 可达 `605~623 MB/s`，仍低于 GPU-only `950 MB/s`。固定 50/50 不是 225 的默认最优策略。
6. 旧错误矩阵中的 GPU-only + CPU 频率点只作为系统耦合参考：它显示 CPU 频率/运行顺序会明显扰动真实 no-ocl 指标（例如 GPU-only `48KB` 在旧矩阵中 no-ocl 压缩从 `422.4` 到 `528.8 MB/s`），但 kernel 指标变化较小。因此后续看频率影响时必须分离 kernel 与真实路径，并避免不相关频率矩阵污染结论。

---

## Nvidia 平台（2026-04-21 Windows / 仅 GPU 默认配置）

### 1. 工件与口径

- 平台：Windows + NVIDIA GeForce RTX 4070 Ti SUPER
- 口径：`--gpu-only` + 默认配置（`A=lzo1x, L=14, BS=64K, LSZ=1`）+ 全样本 25 文件
- 新结果：`lzo-2.10/exp_results/nvidia_rebaseline_20260421/runs/20260421_133348/`
- 对比基线：`lzo-2.10/exp_results/baseline/gpu_only_default/runs/20260414_022428/`
- 汇总工件（本轮新增）：
  - `lzo-2.10/exp_results/nvidia_rebaseline_20260421/analysis/gpu_new_vs_baseline_summary.csv`
  - `lzo-2.10/exp_results/nvidia_rebaseline_20260421/analysis/gpu_new_vs_baseline_filelevel.csv`
  - `lzo-2.10/exp_results/nvidia_rebaseline_20260421/analysis/bar_lzo_gpu_old_vs_new_totals.png`

### 2. 新 GPU 相对旧 GPU 基线（25 文件）

按“逐文件百分比变化”统计（正值=改善）：

| 指标 | 平均 | 中位数 | 正向文件数 | 负向文件数 | 负向最差幅度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CompKernelMBs | +2.94% | +2.46% | 15 | 10 | 9.34% |
| DecKernelMBs | +16.13% | +17.38% | 20 | 5 | 9.45% |
| CompTotalMBs | +98.34% | +98.47% | 25 | 0 | 0.00% |
| DecTotalMBs | +156.76% | +127.85% | 25 | 0 | 0.00% |
| Ratio%（越低越好） | -0.37% | -0.33% | 0 | 25 | 1.42% |

结论：

1. 本轮 LZO 在 NVIDIA 上 total 指标提升非常显著（压缩/解压均全文件正向）；
2. kernel 侧为“多数正向 + 少量回退”结构，仍有继续精修空间；
3. ratio 出现轻微回退（平均 -0.37%），属于本轮需要持续跟踪的负项。

### 3. Intel vs Nvidia 差异原因（本轮归因）

1. **离散显存模型放大 host 路径影响**：NVIDIA 上 total 对 I/O 与回读组装更敏感；
2. **收益来源结构不同**：本轮主要收益集中在 total 链路，说明 host/runtime 调整已成为主导因素；
3. **ratio 与 total 的权衡更明显**：NVIDIA 本轮出现 total 大涨但 ratio 小幅回落，后续需通过参数/内核细化把压缩率拉回。

### 4. 当前结论与方向

- 当前优势：NVIDIA 下 LZO 新实现在 total 吞吐上相对旧基线显著提升；
- 当前短板：ratio 仍有小幅回退（25/25 文件为负向）；
- 后续优先级：保持 total 提升成果，针对 ratio 回退文件做定点排查与参数收敛。
