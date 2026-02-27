# LZO GPU 向量化分析（与当前实现一致）

更新时间：2026-02-27

## 1. 文档目的

本文件仅描述**当前代码中已经实现**并启用的向量化/展开策略，避免与历史实验分支混淆。

涉及实现文件：

- 解压：`lzo_gpu/lzo1x_decomp.cl`、`lzo_gpu/lzo1y_decomp.cl`
- 压缩：`lzo_gpu/lzo1x.cl`、`lzo_gpu/lzo1y.cl`
- 调度：`lzo_gpu/lzo_gpu_core.c`

## 2. 解压向量化：当前真实路径

### 2.1 `UA_COPYN`：分层向量 copy

在 `lzo1x_decomp.cl` / `lzo1y_decomp.cl` 中，`UA_COPYN` 使用固定分层：

1. 32B（两次 `vload16/vstore16`）
2. 16B（`vload16/vstore16`）
3. 8B（`vload8/vstore8`）
4. 4B（`vload4/vstore4`）
5. 1~3B 尾部标量

该路径用于 literal copy 和非重叠 match copy 的主干传输。

### 2.2 `COPY_MATCH`：按 offset 分流

match copy 走 `COPY_MATCH(op, m_pos, len)`，当前实现包含：

1. **非重叠快速返回**：`offset >= len` 时直接 `UA_COPYN`
2. **短 offset 特化**：`offset == 1/2/3/4`，分别用重复模式批量写入
3. **中大 offset 分层向量化**：按 `offset >= 64/32/16/8/4` 做分段批量复制
4. **尾部回落**：剩余字节按标量补齐

### 2.3 解压调用点的“非重叠快路”

在解压主循环中，当前逻辑是：

- `moff = op - m_pos`
- `mlen = t + 2`
- 若 `moff >= mlen`：直接 `UA_COPYN(op, m_pos, mlen)`
- 否则：`COPY_MATCH(op, m_pos, mlen)`

这保证了非重叠常见路径直接走向量 copy，重叠场景保留语义安全。

## 3. 压缩向量化与展开：当前真实路径

### 3.1 `LZO_MEMOPS_COPYN_FAST`

在 `lzo1x.cl` / `lzo1y.cl` 中，压缩侧复制同样采用 `32/16/8/4` 分层向量路径，用于 literal/match copy 辅助函数。

### 3.2 `LZO_USE_UNROLL2 = 1`

当前两套压缩内核都启用了：

- `#define LZO_USE_UNROLL2 1`

作用点是 match-length 扩展热循环（先 16B，再 8B 等），减少循环控制与分支频率，提升 ILP。

### 3.3 词典 epoch 机制（压缩）

压缩词典采用 packed entry + epoch 校验，避免每块清表；该机制与向量 copy 互补：

- 词典路径减轻初始化/写放大
- 向量 copy 降低匹配与输出阶段的访存指令数

## 4. 与本轮实测结果的对应关系

对比 `mod3_lzo_decomp.csv` → `mod6_lzo_full.csv`（GPU common rows = 1512）：

- `CompKernelReported_MBs`：median `+249.38%`
- `DecKernelReported_MBs`：median `+150.91%`
- `Ratio%` median 变化：`0.00`
- 解压正向覆盖率：`1435/1512 = 94.91%`

说明当前向量化 + 展开 + 调度组合已形成稳定主收益路径。

## 5. 已明确移除的历史性描述

以下内容不再作为“当前实现”描述：

1. 旧版文档中基于早期 49 样本的固定结论与带宽上限推断。
2. 未合入主线的“动态 offset<=16 任意模式拼装”方案。
3. 未落地的设备特定预取接口（非标准 OpenCL 路径）。

这些方向可作为实验假设，但不应在实现分析中当作现状。

## 6. 下一步（与当前实现兼容）

1. **主机侧优先**：继续降低 `upload/download/write` 占比，提升端到端吞吐。
2. **尾部数据分型**：针对全零/高重复文件，新增更轻量 match-copy 分流以改善最差样本。
3. **参数自适应**：按块特征调整 `block/local/wi_per_cu`，兼顾中位与尾部。
4. **可观测性补齐**：补充 offset 分布与 path 命中率统计，支持后续定向优化。
