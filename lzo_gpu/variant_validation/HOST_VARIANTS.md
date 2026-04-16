# LZO GPU 主机端变体记录

更新时间：2026-04-16

## 1. 使用约定

- 本文件登记 `lzo_gpu` 的 `host` 组件；
- 正式主判据必须来自 **7~9 次 manual roundtrip** 的分段时间，而不是 `bench` 总吞吐；
- 主机端条目仍需按 `vendor -> stage -> operation` 归类。
- 正式 host 变体只聚焦 **steady-state 主机路径**：`pipeline`、`standard_copy / mapped`、`pack / compaction`；
- `metadata / header` 修补、`telemetry`、bench-only 采样与日志开关，只计为**修正 / hygiene**，不单列为正式 host 变体条目。
- host 变体目录必须附带 **program/source record bundle**：可执行程序、源码文件列表、构建说明、环境开关与补丁摘要缺一不可。
- `hash_table_overhead` 作为特殊轴时，必须在 `buffer_lifecycle` 建账，并与 `KERNEL_VARIANTS.md` 的同轴条目交叉引用。

## 2. NVIDIA

| stage | operation | variant_id | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `buffer_lifecycle` | `hash_table_overhead` | `hash_table_overhead_ab` | `pending` | **[special-axis]** 记录 dictionary/hash table buffer 容量、slot 管理、alloc/reuse 策略对 alloc/upload/download 与显存占用的影响；需同步回写 `KERNEL_VARIANTS.md`。 |
| `host_device_transfer` | `standard_copy_vs_mapped` | `host_transfer_mode_ab` | `pending` | 对比 `LZO_STANDARD_COPY=1` 与 map/unmap 路径，确认 steady-state upload/download/readback 的真实收益。 |
| `dispatch_sync` | `pipeline_overlap` | `host_pipeline_overlap_ab` | `pending` | 只记录会改变 steady-state 提交/重叠路径的 pipeline 变体；bench-only 辅助开关不建正式条目。 |
| `host_feature` | `pack_gate` | `host_pack_gate_ab` | `pending` | 记录 `pack/compaction` 是否启用、启用条件与回退门限；`metadata` / `telemetry` 修正不并入此条。 |

## 3. Intel

| stage | operation | variant_id | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `buffer_lifecycle` | `hash_table_overhead` | `hash_table_overhead_ab` | `pending` | **[special-axis]** 待补 Intel/Linux 侧 dictionary/hash table buffer 容量与管理开销结果，并与 kernel 侧同名条目对位。 |
| `host_device_transfer` | `standard_copy_vs_mapped` | `host_transfer_mode_ab` | `pending` | 待补 Intel/Linux 侧 `standard_copy / mapped` 对位结果。 |
| `dispatch_sync` | `pipeline_overlap` | `host_pipeline_overlap_ab` | `pending` | 待补 Intel/Linux 侧 pipeline steady-state 路径结果。 |
| `host_feature` | `pack_gate` | `host_pack_gate_ab` | `pending` | 待补 Intel/Linux 侧 pack/compaction 门控结果。 |
