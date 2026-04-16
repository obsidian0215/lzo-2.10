# LZO Hybrid 过程阶段目录

更新时间：2026-04-16

## 1. Host 阶段（`host`）

### 1.1 `file_io`

> `metadata / header` 修补默认记为**修正 / hygiene**，除非它直接改变 steady-state 数据路径，否则不作为正式 host 变体主线。

### 1.2 `buffer_lifecycle`

- hash table / dictionary staging buffer 的 alloc / resize / reuse

### 1.3 `host_device_transfer`

- `standard_copy` / `mapped` 传输模式
- packed / sparse / compaction 相关数据搬运

### 1.4 `dispatch_sync`

- pipeline / overlap / steady-state submit order

### 1.5 `host_feature`

- pack / compaction gate
- 其他会改变 steady-state 主机路径的功能开关

> `telemetry`、bench-only 采样/日志、纯修补型 metadata 开关不作为正式 host 变体，统一归为修正记录。
>
> 若 host 侧改动了 hash table / dictionary 的容量、驻留方式或 buffer 生命周期，必须额外打上 `special_axis_tags = ["hash_table_overhead"]`，并交叉引用 `lzo_gpu` 的对应内核条目。

## 2. Scheduler 阶段（`scheduler`）

### 2.1 `work_partition`

- `lzo1x` / `lzo1y` / `D_BITS` 在 CPU/GPU 间如何分配

### 2.2 `execution_model`

- chunk / batch / submit 顺序
- 混合执行流水模型

### 2.3 `fallback_policy`

- fallback threshold
- 模式切换策略

## 3. 分类要求

每个新条目都必须标注：

- `component`: `host` / `scheduler`
- `stage`
- `operation`
- `baseline`
- `motivation`
- `special_axis_tags`
- `program/source record`：host / scheduler 条目必须能回溯到实际程序工件与源码记录
