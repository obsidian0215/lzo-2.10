# LZO GPU 主机端变体账本

更新时间：2026-04-20

## 当前已采纳基线

- 名称：`intel_lzo_gpu_lzo1x_d14_baseline_lock`
- 适用：`lzo1x`, `D_BITS=14`
- 默认 host 路径：Intel 设备走 `mapped/zero-copy`（`LZO_STANDARD_COPY=0`）
- 当前实现要点：
  - grow-only workspace buffer 复用
  - 压缩/解压共用 `standard_copy` / `mapped` 双通路
  - manual 统计中可拆出 `buffer alloc/upload/download` 分段时间
- 证据：`intel/records/host/intel_lzo_gpu_lzo1x_d14_baseline_lock.md`
- 结论：`adopt`

## 当前 Intel 候选

1. `intel_lzo_gpu_lzo1x_d14_standard_copy_r1`
   - 阶段：`host_device_transfer`
   - 操作：`standard_copy_vs_mapped`
   - 动机：对 Intel 平台正式确认“mapped 默认”是否优于强制 `standard copy`。
   - 证据：`intel/records/host/intel_lzo_gpu_lzo1x_d14_standard_copy_r1.md`
  - 当前状态：`reject`

## Intel 已拒绝项

1. `intel_lzo_gpu_lzo1x_d14_standard_copy_r1`
  - 结果：`reject`
  - 原因：`comp_total_no_oci_tp_mbs` 与 `dec_total_no_oci_tp_mbs` 都是 10/10 全负向，`bench_dec_kernel_mbs` 也是 10/10 全负向；Intel 默认继续保持 `mapped/zero-copy`。

## 记账边界

- 仅记录会改变 steady-state 数据路径的 host 变体。
- `metadata/header` 修补、日志、telemetry、bench-only 统计开关默认不作为正式 host 变体。
