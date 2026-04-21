# LZO GPU 内核变体账本

更新时间：2026-04-20

## 当前已采纳基线

### 压缩内核

- 名称：`intel_lzo_gpu_lzo1x_d14_baseline_lock`
- 适用：`lzo1x`, `D_BITS=14`
- 当前默认实现要点：
  - 单 `primary hash`
  - `epoch12 | offset20` 的 32-bit packed dict entry
  - 四位置批量探测与集中写回
  - `match-extension` 的 8/16/32B 展开比较
- 证据：`intel/records/kernel_comp/intel_lzo_gpu_lzo1x_d14_baseline_lock.md`
- 结论：`adopt`

### 解压内核

- 名称：`mainline lzo1x.cl（由 intel_lzo_gpu_lzo1x_d14_short_match_helper_r1 采纳并入）`
- 适用：`lzo1x`, `D_BITS=14`
- 动机：
  - `baseline_lock` 的 decode hot path 已经较强，但 gate-10 与 fullset 都证明：对“短、非重叠 match”额外拉一条极窄 helper，可以稳定提升 decode 吞吐且不伤 ratio。
- 具体操作：
  - 在主线 `lzo1x.cl` 中增加 `lzo1x_fast_direct_match_copy_18()`。
  - 在 `copy_match:` 前置 gate：`if (offset >= mlen && mlen <= 18u)`。
  - 命中时直接做 16B + tail 的 short-match copy；其余路径继续回落到原始 `COPY_MATCH()`。
  - **没有采纳** 被证伪的 D2.1 条件：`mlen >= 8u && mlen <= 18u`。
- 验证结果：
  - gate-10（bench1/manual5）：
    - `bench_dec_kernel_mbs avg=+8.7595%`, `median=+13.3255%`
    - `dec_kernel_tp_mbs avg=+7.8498%`, `median=+8.6113%`
    - `dec_total_no_oci_tp_mbs avg=+3.6027%`, `median=+2.3485%`
  - fullset（bench1/manual5）：
    - `bench_dec_kernel_mbs avg=+10.1537%`, `median=+11.1561%`
    - `dec_kernel_tp_mbs avg=+9.5475%`, `median=+1.2871%`
    - `dec_total_no_oci_tp_mbs avg=+4.9429%`, `median=+2.9542%`
    - `bench_comp_kernel_mbs avg=-0.0284%`
    - `comp_total_no_oci_tp_mbs avg=+0.0531%`, `median=+0.3268%`
    - `ratio_pct = 0.0000%`
- 证据：
  - 主线源码：`lzo_gpu/lzo1x.cl`
  - gate-10 结果：`intel/results/kernel_dec/d2_r1_gate10_b1m5_intel_lzo_gpu_lzo1x_d14_baseline_lock_vs_intel_lzo_gpu_lzo1x_d14_short_match_helper_r1/`
  - fullset 结果：`intel/results/kernel_dec/d2_r1_fullset_b1m5_intel_lzo_gpu_lzo1x_d14_baseline_lock_vs_intel_lzo_gpu_lzo1x_d14_short_match_helper_r1/`
- 结论：`adopt`

## 当前 Intel 候选

当前无需要继续沿用的旧 decode 候选。后续若继续推进 decode，应新开变体编号，不再把已被证伪的 D2.1 或早期 D1 重新包装成“当前候选”。

## Intel 已验证未采纳项

1. `intel_lzo_gpu_lzo1x_d14_copy_match_prune_o4_r1`
   - 结果：`watch / not adopt`
   - 原因：ratio 不变，但 `bench_dec_kernel_mbs avg=-1.6435%`、`manual dec_kernel_tp_mbs avg=-1.3760%`，主指标仍以负向为主；`dec_total_no_oci_tp` 虽均值为正，但中位数为负，未达到采纳标准。
   - 证据：`intel/records/kernel_dec/intel_lzo_gpu_lzo1x_d14_copy_match_prune_o4_r1.md`

2. `intel_lzo_gpu_lzo1x_d14_short_match_helper_len8_18_r3`
   - 结果：`reject`
   - 原因：把 D2 helper 从“`<=18B`”缩窄成“`8..18B`”后，gate-10 `bench1/manual5` 三项 decode 主指标全部翻负：
     - `bench_dec_kernel_mbs avg=-2.4617%`
     - `dec_kernel_tp_mbs avg=-3.9419%`
     - `dec_total_no_oci_tp_mbs avg=-2.0195%`
   - 结论：这不是“更稳的微调”，而是误删了 D2 对 `3..7B` 高频短匹配的有效覆盖。
   - 证据：`intel/results/kernel_dec/d2_micro_gate10_b1_m5_intel_lzo_gpu_lzo1x_d14_baseline_lock_vs_intel_lzo_gpu_lzo1x_d14_short_match_helper_len8_18_r3/`

3. `intel_lzo_gpu_lzo1x_d14_rel_offset_r1`
   - 结果：`reject`
   - 原因：LZ4 M1 风格的 relative-offset loop 不能机械平移到当前 `lzo1x` 解压状态机；历史验证显示它没有形成可采纳的 decode 收益。
   - 证据：`TEMP_LZ4_ADOPTED_MIGRATION_DESIGN_20260421.md`

## 已知低优先级 / 已拒绝方向

- `hash_table_overhead`：当前 `LZO_GPU_REFACTOR_PLAN.md` 已明确记录多条正式 `reject`，包括“少分表 + 少并行”与“保留并行但共享单槽表”的两类路线；后续不再优先尝试。
- `fingerprint_filter`：当前视为待验证机制而非默认路径，且已有历史经验表明热路径额外元数据很容易拖慢压缩吞吐。
- `primary_secondary_hash`：仍可作为独立候选，但不属于当前最优先 Intel 首轮落点；先完成 `copy_match_branch_prune` 与 host 传输轴验证。
