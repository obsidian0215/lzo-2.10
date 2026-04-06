# LZO Hybrid 优化路线（可读版）

## Wave-0 噪声阈值（G1 门禁，2026-04-06）

测量口径

- 样本目录：`/root/samples_subset`
- 重复次数：`3`
- 单轮时长：`bench_seconds=3.5`
- 代表配置：`HYBRID-only, split_layout=prefix, cpu_threads=2, block=64K, local=1, level=15, alg=lzo1x,lzo1y`
- 阈值公式：`|Δ| > max(1.5×MAD, P95(|noise_delta|))`
- 判定方向：`CompTotalMBs`、`DecTotalMBs`、`Ratio%` 均按“越大越好”；`Δ <= -threshold_abs` 记为明显回退，`Δ >= threshold_abs` 记为明显提升。
- 阈值工件：`/root/lzo-2.10/exp_results/noise_profiles/g1/thresholds/lzo_hybrid_fixed.json`、`/root/lzo-2.10/exp_results/noise_profiles/g1/thresholds/lzo_hybrid_adaptive.json`

当前门禁阈值（LZO Hybrid fixed）

- `CompTotalMBs(mean)`：`threshold_abs=19.3598 MB/s`
- `CompTotalMBs(median)`：`threshold_abs=50.6775 MB/s`
- `DecTotalMBs(mean)`：`threshold_abs=14.6347 MB/s`
- `DecTotalMBs(median)`：`threshold_abs=26.6100 MB/s`
- `Ratio%(mean)`：`threshold_abs=0.0000 pctpt`
- `Ratio%(median)`：`threshold_abs=0.0000 pctpt`

当前门禁阈值（LZO Hybrid adaptive）

- `CompTotalMBs(mean)`：`threshold_abs=89.0665 MB/s`
- `CompTotalMBs(median)`：`threshold_abs=91.6845 MB/s`
- `DecTotalMBs(mean)`：`threshold_abs=50.8884 MB/s`
- `DecTotalMBs(median)`：`threshold_abs=226.1200 MB/s`
- `Ratio%(mean)`：`threshold_abs=0.0063 pctpt`
- `Ratio%(median)`：`threshold_abs=0.0495 pctpt`

说明

- 以上阈值用于当前 Wave-0 的 subset/fullset 采纳门禁；若后续切换配置空间或计时口径，需要重新测量并覆盖本节。
- 除 `lzo_hybrid_adaptive` 的 `Ratio%` 外，其余 `Ratio%` 阈值接近 `0`，后续仍按“均值+中位数双判 + 全样本10轮”执行。

## 全集基线结果（当前保留）

- 基线全集目录：`/root/lzo-2.10/exp_results/runs/fullset_allcfg_current_lzo/runs/20260403_140710`
- 基线全集主结果：`lzo_param_sweep.csv`
- 主结果哈希：`sha256=ff8ad767f935253f43d7984e6fceb2c896c0ff1d53cf8d52d73d90764ddbe958`
- 配置汇总哈希：`sha256=be4a4a007a201d9e236ff2c7992d588e62fa02be538a27f86adf41c7fa6dd2ba`
- 行数与完整性：`rows=2300`，`Roundtrip_OK=2300/2300`，引擎覆盖 `CPU/GPU/HYBRID`
- HYBRID 引擎聚合：`CompTotal mean=2781.98 MB/s`，`DecTotal mean=2412.19 MB/s`，`Ratio mean=27.3558%`

## 已采纳修改

### 严格压缩率守护主线（ratio guard v2）

动机

- 先满足“Hybrid 压缩率不低于 CPU/GPU”的硬约束，再在该约束下回收吞吐。

设计

- 在自适应压缩分流上增加 strict ratio guard。
- 引入 ratio-aware 修正项，抑制低比率极端选择。
- 修复非前缀索引映射场景下 GPU upload/readback 一致性。

实现

- 主要文件：`/root/lzo-2.10/lzo_hybrid/lzo_hybrid_core.c`
- 基线工件：`ratio_guard_ab_20260402_002534/artifacts/lzo_hybrid_ab_base`
- 候选工件：`ratio_guard_ab_20260402_002534/artifacts/lzo_hybrid_ab_cand_v2`

测试结果

- fullset：`/root/lzo-2.10/exp_results/runs/ratio_guard_ab_20260402_002534/lzo_hybrid_ab_v2_fullset.json`
- `Comp +20.8622%/-7.7228%`，`Dec +6.7532%/-0.2692%`
- `Ratio +0.244380/+0.565000 pctpt`

采纳原因

- ratio 目标达成，且吞吐具备后续迭代空间，作为主线锚点保留。

### 有界 ratio 搜索（双算法约束）

动机

- adaptive 在双算法上出现极端比率漂移，导致 `Comp/Dec` 同时受损。

设计

- ratio refinement 改为有界搜索，避免无边界 refinement。
- 在压缩与解压路径分别引入算法感知区间约束。

实现

- 文件：`/root/lzo-2.10/lzo_hybrid/lzo_hybrid_core.c`、`/root/lzo-2.10/lzo_hybrid/lzo_hybrid.c`
- 结果目录：`/root/lzo-2.10/exp_results/runs/deep_rework_subset_round3/runs/20260403_121754/`

测试结果

- 相对 fixed:`R0.5`
- `lzo1x: Comp -18.29% -> -6.41%，Dec -16.26% -> -2.85%`
- `lzo1y: Comp -19.72% -> -2.62%，Dec -7.20% -> -1.04%`

采纳原因

- 结构性回退被显著收敛，具备继续细化价值。

### 降低 CPU-only 误触发 + adaptive ratio cache

动机

- 两个主要回退源：小中型文件误触发 CPU-only，以及 bench 循环重复 adaptive 求解开销。

设计

- 下调 `adaptive_skip_ocl_threshold`，减少 `AdaptiveGpuRatio=0`。
- 引入压缩/解压 ratio cache，命中后直接复用决策。

实现

- 文件：`lzo_hybrid.c`、`lzo_hybrid_core.h`、`lzo_hybrid_core.c`
- 运行目录：`/root/lzo-2.10/exp_results/runs/lzh_adaptive_deep_r1/runs/20260403_165528/`

测试结果

- subset：`dComp +6.4270%/+6.1492%`，`dDec +3.0565%/+1.5811%`
- fullset：`dComp +6.6477%/+6.3408%`，`dDec +5.3726%/+1.9963%`
- `AdaptiveGpuRatio=0` 占比：`15% -> 0%`

采纳原因

- subset/fullset 双通过，且直接命中核心回退来源。

## 未采纳修改（表格汇总）

| 修改名（实际语义） | 动机 | 设计与实现 | 测试结果 | 拒绝原因 |
| --- | --- | --- | --- | --- |
| fixed 路径离散化与 adaptive 对齐 | 降低分配抖动 | `split_mode=fixed` 复用 adaptive 调整逻辑（`lzo_hybrid_core.c`） | `R0.7,T2: Comp -3.595%/+0.777%, Dec -14.789%/-30.147%`；`R0.7,T4: Comp -1.437%/-0.974%, Dec -12.014%/-28.342%` | strict 门禁失败 |
| `clFinish` 改 `clWaitForEvents` | 减少全局同步开销 | 队列同步从 finish 改为 event wait（`lzo_hybrid_core.c`） | 多配置均出现负向；如 `R0.3,T2: Comp -1.5634%/-2.5282%` | 无全绿配置 |
| 解压 worker 抢占式调度 | 改善尾部负载不均 | 静态分段改 `atomic_fetch_add` 抢占（`lzo_hybrid_core.c`） | `R0.5,T2: Comp -1.0575%/-1.7538%, Dec +17.2428%/+17.3502%`；`R0.5,T4: Comp -0.6958%/-0.8149%, Dec +31.2716%/+36.6022%` | Dec 提升但 Comp 门禁失败 |
| prefix fastpath 扩展到压缩+解压 | 争取双路径收益 | 双路径统一 fastpath（`lzo_hybrid_core.c`） | 六组配置均有负向；如 `R0.5,T4: Comp -2.0508%/-2.4319%` | strict 失败 |
| event-sync 统一口径复测 | 验证同步改动可复制收益 | 固定 8 文件，baseline/candidate 各 10 轮 | `Comp -2.718%/+2.091%`，`Dec +0.132%/+0.142%` | 压缩均值负向 |
| host-copy/dense-copy 聚合 | 降低 host copy 开销 | 固定参数子集严格 A/B | `Comp -0.0981%/+0.0502%`，`Dec +1.9384%/-0.4948%` | Comp mean 与 Dec median 未过 |
| helper 清理（疑似无用函数删除） | 验证清理是否无损 | 删除多处 helper 并 fullset 复核 | `Comp +0.4824%/-2.0489%`，`Dec +0.0628%/+0.6092%` | 压缩中位数负向 |
| 旧方案全样本复核 | 验证 C03 重测可否转正 | 固定执行目录复测 | `Comp -0.2485%/+0.4060%`，`Dec -0.1977%/+0.1401%` | Comp/Dec 均值负向 |
| 回放步骤一（事件同步迁移） | 验证迁移可复现 | replay1 | `Comp -0.3840%/+0.2373%`，`Dec +0.9113%/-0.0516%` | 有主判负向 |
| 回放步骤二（事件同步迁移） | 同上 | replay2 | `Comp -2.0727%/-2.3454%`，`Dec -0.9725%/+1.3710%` | 双侧波动大且负向 |
| 回放步骤三（事件同步迁移） | 同上 | replay3 | `Comp -1.5470%/+1.4441%`，`Dec -0.6686%/+0.3818%` | 有主判负向 |
| 回放步骤四（事件同步迁移） | 同上 | replay4 | `Comp -0.2429%/-1.3703%`，`Dec -0.4618%/+2.5013%` | 有主判负向 |
| 回放步骤五（事件同步迁移） | 同上 | replay5 | `Comp -0.2816%/-0.7022%`，`Dec -1.0264%/-1.0698%` | 双侧负向 |
| 回放步骤六（事件同步迁移） | 同上 | replay6 | `Comp -2.4104%/-0.0707%`，`Dec -0.7568%/+0.6512%` | 压缩均值显著负向 |
| 解压批量 claim | 降低原子开销 | 解压 worker 批量领取（`host_round_...R5A...`） | 子集：`Comp -0.1896%/-0.2222%`，全集：`Comp -0.1003%/-2.0297%`（Dec 虽正） | Comp 持续负向 |
| 前缀布局直写（免索引） | 复制 L4H 的 host 优化 | 压缩索引组织精简（`...R5B...FULLSET...`） | `Comp -1.1855%/-0.7784%`，`Dec +0.1345%/+0.3650%` | 压缩双负向 |
| 最小化解压 offsets 构建 | 降低准备开销 | 仅 CPU 分段做必要 offsets | `Comp +0.1819%/+0.2102%`，`Dec -0.1337%/+0.6448%` | Dec mean 负向 |
| 压缩批量 claim | 减少原子热点 | 压缩线程批量领取 | `Comp -0.1822%/-0.0697%`，`Dec +0.1084%/+0.2461%` | 压缩未过门禁 |
| 解压调度强化（大块优先） | 优先提升 Dec | 大块场景增强解压调度 | `Comp -0.2041%/-0.7057%`，`Dec +15.6184%/+15.6794%` | Comp 门禁失败 |
| adaptive 多目标权重初版 | 统一性能/能效/压缩率 | 动态权重 + ratio 软约束 | `Comp -0.9371%/-1.3114%`，`Dec -3.3846%/-0.9763%` | Comp/Dec 双负向 |
| adaptive 保守约束修复版 | 试图修复初版双负向 | 小文件 CPU-only + 吞吐比约束 | `Comp +4.5925%/-9.1769%`，`Dec +1.1149%/+3.5716%` | Comp median 严重负向 |
| adaptive 仅保留 ratio guard | 降低干预强度 | 只保留温和 ratio guard | `Comp +4.6302%/-9.6207%`，`Dec -2.3510%/-3.4235%` | Dec 双负向 |
| adaptive 只调解压惩罚项 | 保 Dec 并控副作用 | 仅改 `dec_host_penalty` | 子集：正向；全集：`Comp -1.9873%/-0.3946%` | 全集压缩失败 |
| adaptive 惩罚项中值回调 | 平衡 R4 副作用 | `dec_host_penalty` 5.0->3.5 | `Comp -2.1438%/-0.7153%`，`Dec +0.0169%/-6.8963%` | Comp 与 Dec median 失败 |
| 解压 metadata 哈希缓存 | 减少重复上传 | metadata 不变则跳过上传（`hybrid_meta_cache_r1`） | `dComp -0.5982%/-0.4900%`，`dDec +0.3571%/-0.1356%` | Comp 双负 + Dec median 负向 |

## 当前代码一致性检查结论

- `lzo_hybrid_core.h` 保留采纳项所需 `adaptive_ratio_cache_comp/dec` 字段。
- `lzo_hybrid_core.c` 保留压缩/解压缓存命中快路。
- 已拒绝的“解压 metadata 哈希缓存”逻辑未保留在主线。
