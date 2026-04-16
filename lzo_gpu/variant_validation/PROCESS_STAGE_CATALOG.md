# LZO GPU 过程阶段目录

更新时间：2026-04-16

## 1. 作用

本文件用于把 `lzo_gpu` 的候选按 **组件 / 阶段 / 操作** 建账，避免把 `D_BITS`、hash、copy-match、主机通信之类的变化混在一起。

## 2. 压缩内核阶段（`kernel_comp`）

### 2.1 `hash_dict`

对应过程：

- `DINDEX()` / `dict_load32()` / `dict_store32()`
- 主 hash、备 hash、fingerprint、packed-entry 布局
- `D_BITS=13/14/15` 下的字典命中策略
- hash table / dictionary 容量、slot 数与管理开销

典型操作：

- `primary_hash`
- `primary_secondary_hash`
- `fingerprint_filter`
- `dict_entry_layout`
- `hash_table_overhead`

### 2.2 `search_probe`

对应过程：

- `literal -> next -> next_slow` 主搜索循环
- 向量化批量 hash 读取
- `ip += 1 + ((ip - ii) >> 5)` 步长增长

典型操作：

- `vector_probe_batch`
- `step_policy`
- `insert_timing`

### 2.3 `match_count`

对应过程：

- `m_len` 统计循环
- `8B/16B/32B` 批量比较
- `ctz()` 驱动的尾部收敛

典型操作：

- `batch_width`
- `unroll_policy`
- `short_match_gate`

### 2.4 `literal_emit`

对应过程：

- literal 长度编码
- `LZO_MEMOPS_COPYN_FAST()` 拷贝路径
- `t <= 3 / <= 18 / > 18` 编码分支

典型操作：

- `literal_copy_dispatch`
- `length_emit`

### 2.5 `match_emit`

对应过程：

- `M2/M3/M4` 编码分派
- offset 归类与 marker 写出
- terminate block 路径

典型操作：

- `offset_bucket_policy`
- `marker_emit`
- `terminate_path`

## 3. 解压内核阶段（`kernel_dec`）

### 3.1 `token_decode`

对应过程：

- literal/match token 解析
- `M2/M3/M4` marker 识别
- EOF/尾部识别

### 3.2 `literal_copy`

对应过程：

- literal run copy
- `UA_COPYN()` 的非重叠复制路径

### 3.3 `match_copy`

对应过程：

- `COPY_MATCH()`
- 小 offset 特化
- overlap forward-copy 与向量 ladder

### 3.4 `tail_boundary`

对应过程：

- `eof_found`
- block 尾部边界与输出长度

## 4. 主机端阶段（`host`）

### 4.1 `file_io`

> `metadata / header` 修补默认记为**修正 / hygiene**，除非它直接改变 steady-state 数据路径，否则不作为正式 host 变体主线。

### 4.2 `buffer_lifecycle`

- dictionary / hash table buffer 的 alloc / resize / reuse

### 4.3 `host_device_transfer`

- `standard_copy` / `mapped` 传输模式
- packed / sparse / compaction 相关数据搬运

### 4.4 `dispatch_sync`

- pipeline / overlap / steady-state submit order

### 4.5 `host_feature`

> 含义与 `lz4_gpu` 相同：分别用于记录文件路径、buffer 生命周期、上传下载、事件/队列同步，以及会改变 steady-state 主机路径的 pack/compaction 类特性。
> `telemetry`、bench-only 采样/日志、纯修补型 metadata 开关不作为正式 host 变体，统一归为修正记录。
>
> 凡是改变 dictionary/hash table 容量、slot 数量、padding/reuse 策略的候选，必须额外打上 `special_axis_tags = ["hash_table_overhead"]`，并同步写入 kernel + host 两侧记录。

## 5. 分类要求

每个新条目都必须至少写明：

- `component`: `kernel` / `host`
- `optimization_object`: `kernel_comp` / `kernel_dec` / `host`
- `stage`
- `operation`
- `baseline`
- `motivation`
- `special_axis_tags`：若属于 `hash_table_overhead` 之类跨组件特殊轴，必须显式标记
