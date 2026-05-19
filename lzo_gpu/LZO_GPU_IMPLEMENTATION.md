# LZO GPU 当前实现说明

更新时间：2026-05-19

本文只描述 `lzo_gpu` 当前主线已经启用的实现。历史诊断、拒绝项和路线讨论放在 `lzo_experimental`；只有验证后采纳的改动才写入本文。

## 1. 当前主线范围

当前实现覆盖：

- `lzo_gpu` standalone：单次真实压缩、解压、bench；
- `lzo_gpu_daemon`：常驻 OpenCL context/queue/program/kernel/workspace；
- `lzo_gpu_client`：路径请求或 `--raw-buffer` 标准输入/输出请求；
- `lzo1x` / `lzo1y` 两套内核。

当前已采纳改动：

| 类别 | 已采纳改动 | 作用 |
|---|---|---|
| 压缩字典 | `block_size <= 64KB && D_BITS <= 16` 启用 `LZO_DICT_U16_CLEAR=1` | 字典 entry 从 32-bit 降到 16-bit，减少 dict pool、随机读写和 cache 压力 |
| 字典布局 | kernel 内 `dict_stride` 按 entry byte width 计算 | 16-bit 模式真实压缩表间距，不再只是死分支 |
| 编译路径 | `lzo_load_comp_kernel_for_block()` 按 block size 构建压缩 program | standalone/bench 能正确选择 clear16 或 epoch32 |
| daemon 缓存 | daemon 压缩 program/kernel 按 `alg × D_BITS × dict_mode` 缓存 | 避免 16-bit/32-bit 字典模式共用错误 program |
| 解压内核 | token `post_lit` fast path | match 完成后不再统一回读 `ip[-2] & 3` |
| 解压 copy | `<=18B` 非重叠 direct match helper，且 `<=8B` 用单次 8B copy | 降低高频短 match 的无效 16B 读写 |
| 真实输出 | 解压 `out_lens` 条件化、chunked readback/write 平台化 | 真实文件路径减少无用写回和大输出尾部开销 |
| daemon 协议 | request magic/version 与 `RAW_BUFFER` | 防止新旧协议混用，并支持 raw-buffer bridge |

未采纳或不属于当前主线：

- LZOG2 多阶段重写；
- 压缩 gather/compaction/pipeline；
- `sig4/sig8` 指纹过滤；
- 文件特征 gate；
- v2 hybrid/range 调度层默认化。

## 2. 压缩内核

### 2.1 调度模型

标准入口：

- `lzo1x_block_compress`
- `lzo1y_block_compress`

每个 work-item 绑定一个 dict slot，并处理一个或多个 block：

```c
dict_slot = wi % dict_pool_size;
for each assigned block:
    dict_clear_for_block(dict);
    do_compress(..., dict, epoch);
```

host 侧 `lzo_compress_core()` 先确定 block size 与 block 数，再按资源限制决定 launch：

```text
target_items = nblk
occ_cap      = max(1024, CU * LZO_OCC_FACTOR_DEFAULT)
mem_cap      = min(global_mem/6, max_alloc*0.9) / dict_per_block
target_items = min(target_items, occ_cap, mem_cap)
global_size  = round_up(target_items, local_size)
pool_size    = global_size
```

当前仍是“每个 active WI 一个 dict slot”。这不是最终 state-slot 解耦设计，但配合 16-bit entry 后，默认 64KB 及以下 block 的 dict pool 压力已减半。

### 2.2 字典 entry 模式

`lzo_dict_u16_clear_for_block(block_size, bits)` 决定压缩 program 的字典模式：

```text
block_size <= 64KB 且 D_BITS <= 16: clear16
其他情况: epoch32
```

`clear16`：

- build flag：`-D LZO_DICT_U16_CLEAR=1`
- entry 类型：`ushort`
- entry 内容：block-local offset；
- empty 值：`0`
- 每 block 进入压缩前清表；
- dict size：`(1 << D_BITS) * 2B`
- `D_BITS=14` 时每 state 约 `32KB`。

`epoch32`：

- build flag：`-D LZO_DICT_U16_CLEAR=0`
- entry 类型：`uint`
- entry 内容：`12-bit epoch | 20-bit offset`
- 普通 block 不清整表，靠 epoch 判定 stale；
- dict size：`(1 << D_BITS) * 4B`
- `D_BITS=14` 时每 state 约 `64KB`。

16-bit 模式之所以安全，是因为当前主线默认 block 不超过 64KB，block-local offset 可由 16 bit 表示。这个改动不减少 hash slot 数，因此相比降低 `D_BITS`，它不应引入额外碰撞，也不应改变压缩率。

### 2.3 candidate 与 probe hot path

当前压缩仍是单 primary hash：

1. 读取当前位置附近连续字节；
2. 生成 4 个相邻位置 hash；
3. 批量读取 4 个 dict entry；
4. 做 epoch/empty、距离和 3/4 字节候选验证；
5. 集中写回 dict；
6. 命中后进入 match extension 和 token output。

这条路径的核心成本是：

- dict 随机 load/store；
- 候选字节读取；
- failed compare；
- match extension；
- literal/match token 写出。

因此后续压缩优化必须证明减少这些真实操作之一。只改代码形状、加无证据 gate 或增加指纹字段都不进入主线。

### 2.4 match extension 与宽比较

`match extension` 已经使用 8B/32B 等展开比较，失配时用 bit 操作定位首个不同字节。宽比较不能继续无条件加宽：

- 如果 match 很短，更宽比较只会增加额外 load；
- 如果引入更多 tail/越界分支，会增加寄存器和分支压力；
- 只有诊断证明长 match 占主导时，才允许作为独立变体验证。

## 3. 解压内核

标准入口：

- `lzo1x_block_decompress`
- `lzo1y_block_decompress`

解压不需要大字典池，主要成本在 token decode、literal copy、match copy 和输出回收。

当前采纳两项内核改动：

1. `post_lit`：M2/M3/M4 解析时保存后继 literal 长度，`match_done` 不再统一回读输入。
2. `short8 direct-copy`：当 `offset >= mlen && mlen <= 18` 时进入 direct helper；`mlen <= 8` 只做一次 `vload8/vstore8`，`9..18` 保留 16B 路径。

`COPY_MATCH()` 仍保留分层语义：

- `offset >= len`：非重叠向量 copy；
- `offset == 1/2/4`：重复模式专门复制；
- 其他 offset：按 `64/32/16/8/4B` 分层；
- overlap match 不能用普通宽 copy 替代。

## 4. Host 路径

### 4.1 program/kernel 加载

压缩路径必须使用：

```c
lzo_load_comp_kernel_for_block(ctx, dev, alg, level, block_size, ...)
```

该函数最终调用：

```c
lzo_load_program_with_dbits_and_block(..., block_size, ...)
```

并在 build options 中加入：

```text
-D D_BITS=<level>
-D LZO_DICT_U16_CLEAR=<0|1>
```

旧 `lzo_load_comp_kernel()` 仍保留为兼容 wrapper，但无法表达 block-size 相关字典模式，不应作为新增压缩调用的入口。

解压 program 不依赖压缩字典模式，仍可用 `lzo_load_program_with_dbits()`。

### 4.2 workspace 与字典池

`lzo_compress_core()` 用 `lzo_dict_entry_bytes_for_block()` 计算 dict pool：

```text
dict_per_block = (1 << D_BITS) * entry_bytes
total_dict_size = pool_size * dict_per_block
```

`core_get_or_create_buffer()` 仍是 grow-only 复用，只在容量不足时扩容。16-bit clear 降低的是压缩字典池大小与热路径 entry 读写宽度，不改变 output buffer、length table 或压缩格式。

### 4.3 memory copy 与真实输出

Intel/统一内存平台默认 mapped/zero-copy；standard copy 只作为兼容和对照路径。

真实解压路径支持：

- 默认不写无用 `out_lens`；
- Windows 大输出可启用 chunked readback/write；
- Linux 默认保守关闭 chunked，避免 host path 退化。

### 4.4 daemon

daemon 复用：

- OpenCL context；
- worker queue；
- worker workspace；
- 压缩/解压 kernel；
- program cache。

压缩 program/kernel 缓存维度是：

```text
alg × D_BITS × dict_mode
```

其中 `dict_mode=1` 表示 `clear16`，`dict_mode=0` 表示 `epoch32`。这避免同一个 `D_BITS` 在 64KB 与大 block 之间错误复用 program。

`--raw-buffer` 协议：

1. client 从 stdin 读取 payload；
2. request 设置 `LZO_DAEMON_FLAG_RAW_BUFFER`；
3. daemon 用 `memfd` 把 payload 暴露给现有文件路径 core；
4. 输出长度和 payload 通过 socket 回传。

这条路径用于 socket/pipe bridge。它复用现有 core，风险低；如果后续小消息场景 memfd 成为瓶颈，再考虑真正 buffer-core。

## 5. bench 脚本口径

`tools/bench_lzo.py` 与 LZ4 bench 保持统一输出：

- `raw.csv`
- `per_file_summary.csv`
- `aggregate.csv`

默认：

```text
bench:  1 轮，每轮 5 秒
manual: 6 轮真实压缩/解压
```

manual 产物写入统一临时目录，测试后删除，不污染样本目录。功耗/频率由外部 wrapper 采集，不写入 bench 主 CSV。

## 6. 105 远端验证结果

测试平台：`192.168.2.105`，Intel Iris Xe OpenCL GPU，样本目录 `/root/samples`，共 26 个文件。

测试口径：

- 当前实现：`/root/lzo-2.10/lzo_gpu/lzo_gpu`
- 对照基线：`/root/lzo-2.10/lzo_gpu_baseline_/lzo_gpu`
- 配置：`lzo1x`，`D_BITS=14/15`，`48KB/64KB`，`local=1`
- 轮次：每配置 `1` 轮 bench，每文件 `6` 轮真实压缩/解压 manual
- 结果归档：`exp_results/remote105_gpu_compare_full6`

总体结果：

| 配置 | ratio 中位数 | 压缩 kernel 中位 | 解压 kernel 中位 | 端到端压缩中位 | 端到端解压中位 |
|---|---:|---:|---:|---:|---:|
| baseline D14 48KB | 32.390% | 1152.33 MB/s | 2907.74 MB/s | 606.73 MB/s | 1025.84 MB/s |
| current D14 48KB | 32.390% | 1141.11 MB/s | 2922.30 MB/s | 630.14 MB/s | 1025.54 MB/s |
| baseline D14 64KB | 32.215% | 947.96 MB/s | 2483.90 MB/s | 561.93 MB/s | 990.92 MB/s |
| current D14 64KB | 32.215% | 941.35 MB/s | 2488.31 MB/s | 574.04 MB/s | 997.29 MB/s |
| baseline D15 48KB | 32.345% | 1117.65 MB/s | 2902.83 MB/s | 541.13 MB/s | 1021.45 MB/s |
| current D15 48KB | 32.345% | 1087.73 MB/s | 2903.40 MB/s | 586.25 MB/s | 1021.12 MB/s |
| baseline D15 64KB | 32.160% | 922.58 MB/s | 2461.13 MB/s | 509.88 MB/s | 989.37 MB/s |
| current D15 64KB | 32.160% | 913.62 MB/s | 2464.52 MB/s | 552.46 MB/s | 985.43 MB/s |

文件级相对基线统计：

- 压缩率：全部配置中位差异 `0.00%`，说明 `clear16` 不改变 LZO 输出格式或候选表达能力。
- 压缩 kernel：中位 `-1.05%`，平均 `-0.73%`，说明 105 上 `clear16 + 清表` 没有带来核心压缩内核收益。
- 解压 kernel：中位 `+0.09%`，平均 `+0.07%`，属于基本持平。
- 端到端压缩：中位 `+4.25%`，平均 `+5.51%`，主要来自字典池变小后 host/device buffer 与真实路径开销降低。
- 端到端解压：中位 `-0.31%`，平均 `-0.25%`，基本持平，Linux 默认不启用 chunked。

结论：当前 LZO 采纳项对压缩率安全，且能降低真实压缩路径开销；但它不是压缩 kernel 本体优化。后续 LZO 压缩 kernel 要继续围绕 probe hot path 和 state-slot 解耦，而不能把端到端收益误判为核心 kernel 已优化。

最终 GPU 基线已在同一机器重新收集，结果归档到 `exp_results/gpu_baseline_105_final`。当前基线为：

| 配置 | ratio 中位数 | 压缩 kernel 中位 | 解压 kernel 中位 | 端到端压缩中位 | 端到端解压中位 |
|---|---:|---:|---:|---:|---:|
| D14 48KB | 32.390% | 1137.36 MB/s | 2921.97 MB/s | 623.41 MB/s | 998.89 MB/s |
| D14 64KB | 32.215% | 936.77 MB/s | 2490.94 MB/s | 571.60 MB/s | 968.12 MB/s |
| D15 48KB | 32.345% | 1079.73 MB/s | 2906.95 MB/s | 579.70 MB/s | 993.01 MB/s |
| D15 64KB | 32.160% | 917.20 MB/s | 2476.28 MB/s | 548.17 MB/s | 963.54 MB/s |

## 7. 后续优化边界

下一轮优先级：

1. state-slot 与并发解耦：降低 dict pool，不牺牲 occupancy；
2. probe hot path：减少随机 load、candidate read、failed compare 或 store；
3. 解压分支简化：只做能减少高频分支的改动；
4. daemon buffer-core：只在 raw-buffer/memfd 成为端到端瓶颈时推进。

禁止重新发散：

- LZOG2；
- bench-only gather/compaction；
- pipeline/overlap；
- sig4/sig8；
- 无证据文件特征 gate；
- 未经分布诊断的更宽 compare/copy。
