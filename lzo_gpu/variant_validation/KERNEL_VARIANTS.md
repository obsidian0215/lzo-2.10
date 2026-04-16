# LZO GPU 内核变体记录

更新时间：2026-04-16

## 1. 使用约定

- 本文件登记 `lzo_gpu` 的内核组件，覆盖 `kernel_comp` 与 `kernel_dec`；
- 记录组织方式固定为：`vendor -> optimization_object -> stage -> operation`；
- 对 `lzo1x` / `lzo1y` / `D_BITS=13/14/15`，只要属于同一机制，也优先复用同一个 `variant_id`，把算法与级别差异写到结果里，而不是额外发明一套名字。
- 若变体属于 `hash_table_overhead` 这类跨组件特殊轴，必须与 `HOST_VARIANTS.md` 同步建账，并在两边使用同一个 special-axis 标记。

## 2. 当前默认源码基线（2026-04-16 收敛）

- `lzo1x.cl` / `lzo1y.cl` 已把 **主备 hash / secondary hash probe** 从默认慢路径中移除；
- 默认压缩路径回收到 **单 primary hash** 基线；
- 默认解压路径已去掉会增加分支深度、但没有额外向量收益的 `COPY_MATCH(offset == 3)` 专门分支；
- `copy_match` 调用点已统一收敛为直接走 `COPY_MATCH()`，不再在调用点额外分叉 `UA_COPYN vs COPY_MATCH`；
- 这类“源码收敛”属于 **baseline cleanup**；其中 `kernel_dec / match_copy / copy_match_branch_prune` 已完成稳定复核并可随默认源码一并采纳；
- `fingerprint` 与 `主备 hash / dual probe` 不再留在默认实现中，而是作为单独变体继续建账与验证；
- `simplified_current` 相对 `baseline_pre_simplify` 的稳定复核已完成：`AvgBench +1.37%`、`AvgBenchDec -0.06%`、`AvgCompKernel +0.41%`、`AvgDecKernel -0.55%`，`Bench/Comp ratio delta = 0`；
- 负向尾部分别为 `Bench -2.72%`、`BenchDec -7.18%`、`Comp -5.16%`、`Dec -18.83%`；其中 `Bench/Comp` 侧仍落在同场 `control_a/control_b` 噪声包络内，`DecKernel` 个别样本尾部偏深，但未形成系统性均值退化；
- 结论：当前简化后源码可正式接替 `pre_simplify` 复核包，作为新的默认源码基线。

## 3. NVIDIA

### 3.1 `kernel_comp / hash_dict / fingerprint`

#### `fp_filter_pending`

- 显示名称：`Fingerprint 过滤（待验证）`
- 状态：`pending`
- 适用算法：`lzo1x` / `lzo1y`
- 动机：在 primary hash 保持单路的前提下，尝试用轻量 fingerprint 过滤掉一部分 false match compare，观察其是否能覆盖 packed-entry 解码成本。
- 设计与实现：建议以 `epoch + fp + low position` 的 packed-entry 形式落到专用候选目录，并分别测试 `fp4/fp8` 或开/关版本。
- 整体结果：当前默认源码未保留该机制，正式结果尚未建账。
- 判定：`pending`
- 证据：`lzo-2.10/lzo_gpu/lzo1x.cl`、`lzo-2.10/lzo_gpu/lzo1y.cl`

### 3.2 `kernel_comp / hash_dict / primary_secondary_hash`

#### `dual_hash_probe_pending`

- 显示名称：`主备 hash / dual probe（待验证）`
- 状态：`pending`
- 适用算法：`lzo1x` / `lzo1y`
- 动机：历史默认源码里曾在 slow path 同时探测主 hash 与备 hash；现在先把它移出默认基线，再独立验证它是否真的值得保留。
- 设计与实现：候选版本应在同一 `sequence` 上计算 `DINDEX` 与 `DINDEX_ALT`，并明确记录 store policy、命中优先级与 `D_BITS` 固定值。
- 整体结果：默认源码移除该机制后的 `simplified_current` 已完成稳定复核并可作为新基线；但把 dual probe 作为独立候选重新引入时，仍未形成正式 benchmark / manual 建账。
- 判定：`pending`
- 证据：
  - `lzo-2.10/lzo_gpu/lzo1x.cl`
  - `lzo-2.10/lzo_gpu/lzo1y.cl`
  - `lzo-2.10/lzo_gpu/variant_validation/nvidia/results/lzo_simplified_current_stable_recheck_20260415/summary_comparison.txt`

### 3.2B `kernel_comp / hash_dict / hash_table_overhead`

#### `hash_table_overhead_ab`

- 显示名称：`Dictionary / hash table 容量与管理开销`
- 状态：`pending`
- special-axis：`hash_table_overhead`
- 适用算法：`lzo1x` / `lzo1y`
- 动机：当前 dictionary/hash table 全局内存按 active work-item slot 扩张；需要把容量、slot 布局、padding/reuse 策略对 kernel 吞吐与 host buffer 开销的影响正式建账。
- 设计与实现：在保持 `block=64KB`、`localsize=1`、`level=14` 不变的前提下，只扫描 dictionary/hash table 容量、entry 布局或管理逻辑；host 侧同步记录 alloc/upload/download 段。
- 整体结果：尚未形成正式 A/B 结果，当前先作为特殊轴占位。
- 判定：`pending`
- 证据：
  - `lzo-2.10/lzo_gpu/lzo1x.cl`
  - `lzo-2.10/lzo_gpu/lzo1y.cl`
  - `lzo-2.10/lzo_gpu/variant_validation/HOST_VARIANTS.md`

### 3.3 `kernel_dec / match_copy / copy_match_branch_prune`

| variant_id | 状态 | 整体结果与判定 | 证据 |
| --- | --- | --- | --- |
| `copy_match_branch_prune_baseline_20260416` | `adopt` | 以 `baseline_pre_simplify` 为锚点的稳定复核表明：简化后源码 `ratio` 完全不变，`AvgBench +1.37%`、`AvgCompKernel +0.41%`；解压均值基本持平（`AvgBenchDec -0.06%`、`AvgDecKernel -0.55%`）。负向尾部里 `Bench/Comp` 仍落在同场 control 噪声包络内，`DecKernel` 最差样本 `-18.83%` 偏深但未形成系统性退化，因此该分支裁剪随当前默认源码一起收敛为新基线。 | `lzo-2.10/lzo_gpu/variant_validation/nvidia/results/lzo_simplified_current_stable_recheck_20260415/summary_comparison.txt`；`lzo-2.10/lzo_gpu/lzo1x.cl`；`lzo-2.10/lzo_gpu/lzo1y.cl` |

## 4. Intel

> Intel/Linux 侧当前还没有正式结果，先保留同名条目，后续直接在本章补平台差异。

| stage | operation | 代表变体 | 状态 |
| --- | --- | --- | --- |
| `hash_dict` | `fingerprint_filter` | `fp_filter_pending` | `pending` |
| `hash_dict` | `primary_secondary_hash` | `dual_hash_probe_pending` | `pending` |
| `hash_dict` | `hash_table_overhead` | `hash_table_overhead_ab` | `pending` |
| `match_copy` | `copy_match_branch_prune` | `copy_match_branch_prune_baseline_20260416` | `pending` |
