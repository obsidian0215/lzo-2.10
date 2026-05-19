# LZO GPU / Hybrid：相关工作对比、现状与后续重构路线

本文基于 `C:\Users\obsid\Desktop\博士毕设\edgecompress\relate_works.md`、当前 `lzo_gpu/lzo_hybrid` 实现，以及近期对字典、并发、block size 和 GPU 压缩瓶颈的讨论，重新整理 LZO GPU/hybrid 的技术位置与后续路线。

当前优先级固定为：

1. **字典/state-slot 与并发结构**；
2. **适配新字典结构的压缩 kernel**；
3. **解压 kernel 与输出排水**；
4. **hybrid CPU/GPU 比例调度**；
5. **daemon / 边缘信息流验证**。

边缘信息流是最终系统验证层，不是当前核心内核优化的先导。压缩 kernel 的调度分配必须服从字典设计；否则会继续围绕旧 per-work-item dict 做局部优化，后续被 state-slot 重构推翻。

---

## 1. 相关工作与参数对比

### 1.1 GPU LZSS / GPULZ 类工作

`CULZSS`、pipeline LZSS 和 `GPULZ` 的共同结论是：GPU 上不能直接照搬串行 LZ 状态机，而要通过 block/chunk 独立化降低依赖。

`GPULZ` 的参数很有参考价值：

- 数据被分成 block，再分成 chunk；
- chunk 映射到 GPU thread block；
- match 不跨 chunk；
- 扫描参数包括：
  - chunk size：`2048 / 4096 / 8192 / 16384`；
  - sliding window：`32 / 64 / 128 / 255`；
  - symbol length：`1 / 2 / 4`；
- 默认配置偏向小 chunk 和短窗口，例如 `chunk=2048`、`window=128`、`symbol=2`。

含义：

- 并行 GPU LZ 类算法经常主动缩短局部依赖范围，而不是保留 CPU 格式的最大历史窗口；
- 小 chunk 提高并发和吞吐，但会损失跨 chunk match；
- 大 chunk 提高压缩率，但会降低并行度和吞吐；
- 这类工作用“窗口缩短 + chunk 并行”换吞吐，但它们通常不是严格保持 LZO/LZ4 格式和压缩率的路线。

对 `lzo_gpu` 的启发：

- 可以借鉴“局部状态 + 高并发”的结构；
- 不能直接采用 `window=128` 这类激进短窗口，因为 LZO 目标仍要求压缩率接近原始实现；
- 应优先控制 **物理字典 state 数**，而不是先缩短 LZO 格式历史。

### 1.2 LZ4/LZO FPGA/ASIC 加速工作

FPGA/ASIC LZ4 工作通常会：

- 限制并行窗口；
- 降低 hash table 复杂度；
- 限制每个窗口内处理的 match 数；
- 用更小或更可控的硬件表换吞吐和频率。

这些工作说明“缩表/限制候选”是硬件化常用手段，但代价是压缩率下降。已有硬件工作中，缩表或限制匹配可能带来数个百分点甚至更高的压缩率损失；这不能无条件迁移到 `lzo_gpu`。

对 `lzo_gpu` 的启发：

- 资源预算必须是一等设计目标；
- hash/dict table 不应随 work-item 数无限膨胀；
- 但压缩率约束比纯硬件吞吐设计更严格，缩字典要在全样本上量化。

### 1.3 nvCOMP / batched GPU compression

nvCOMP LZ4 等 GPU 压缩库采用 batched/chunk 方式，把大输入拆成多个独立 chunk。典型 chunk 粒度经常围绕 `64KB`，因为 LZ4 格式最大 match distance 是 `65535`。

对 LZO 的意义：

- GPU 压缩库普遍依赖 chunk/batch 并行；
- chunk 边界压缩率损失是已知成本；
- 真正工程问题是找到吞吐、压缩率和资源占用之间的稳定点。

### 1.4 LZO 原始实现与当前 GPU 参数

LZO1X/LZO1Y 的格式历史上限不是无限长：

- `lzo1x`：`M4_MAX_OFFSET = 0xbfff = 49151`，见 `lzo1x.cl`；
- `lzo1y`：`M4_MAX_OFFSET = 0xbfff = 49151`，见 `lzo1y.cl`；
- 原始 `lzo1x_1` 工作内存通常是 `16384 * sizeof(dict)`；
- 当前 GPU 默认 `D_BITS=14`，即每个 state `16384` 个 `uint` entry，约 `64KB/state`；
- `D_BITS=13` 是 `32KB/state`；
- `D_BITS=15` 是 `128KB/state`。

当前 GPU 的关键问题不是单个 `D_BITS=14` 不合理，而是 **state 数量与 active work-item 绑定**。如果 active work-item 多，字典池总大小线性放大。

---

## 2. 块增多造成压缩率损失的来源

块数增加带来两类损失：

1. 资源和调度损失：字典池变大、buffer 管理变重、kernel/host 元数据更多；
2. 压缩率损失：block 边界切断历史、字典冷启动、候选质量下降。

本节重点解释压缩率损失。

### 2.1 跨 block 历史丢失

LZO 的 match 不能跨当前 GPU block。原始串行 LZO 可以在当前偏移向前最多约 `48KB` 找 match；GPU 分块后，每个 block 的起点历史被清空。

理论上，第 `i` 个 block 内偏移 `p` 处可用的本 block 历史长度是 `p`，而原始串行路径最多可用 `W`，其中：

```text
W ≈ 49151 bytes
B = block_size
available_history(p) = min(p, W)
lost_history(p) = max(0, W - p)
```

当 `B <= W` 时，block 内所有位置都处于“历史不足”状态；平均可用历史约为 `B/2`，相对完整窗口的平均覆盖为：

```text
avg_history_ratio ≈ B / (2W)
```

例如：

- `B=32KB`，`W≈48KB`：平均历史覆盖约 `33%`；
- `B=48KB`，`W≈48KB`：平均历史覆盖约 `50%`；
- `B=64KB`，`W≈48KB`：前 `48KB` 逐渐 warm-up，后 `16KB` 才拥有完整窗口，平均历史覆盖约 `62.5%`。

这不是直接等于压缩率损失，但说明小 block 对长距离重复更不友好。

### 2.2 字典冷启动

每个 block 开始时字典为空，短时间内 match 候选少。冷启动损失集中在：

- block 前部；
- 重复结构跨 block 边界时；
- 文本、内存镜像、日志、结构化状态等存在长距离重复的数据。

对 x-ray 等难压缩数据，冷启动影响可能很小，因为本来候选质量就低；不能用这类样本作为主要压缩率判断依据。

### 2.3 边界处 match 被拆成 literal

如果原始串行压缩会把某段编码成跨 block match，GPU 分块后只能把它编码为 literal 或较短 match。

其贡献可以通过边界损失估计器定量：

```text
boundary_loss_bytes =
  原始串行/参考压缩中 match_source < block_start 且 match_pos >= block_start 的 match 覆盖字节数
```

这个值除以原始输入大小，可以估算“仅由跨 block match 禁止导致的潜在压缩率损失上界”。

### 2.4 header / length table 开销

block 越多，容器元数据越多。LZO GPU 容器至少包含：

- 文件头；
- 每 block 压缩长度表；
- 每 block payload。

如果每 block length 是 4 字节，那么元数据开销近似：

```text
overhead_ratio ≈ 4 * nblk / input_size ≈ 4 / B
```

量级：

- `B=32KB`：约 `0.012%`；
- `B=48KB`：约 `0.008%`；
- `B=64KB`：约 `0.006%`。

所以 header 不是主要压缩率损失来源，真正主要的是跨 block history 和冷启动。

### 2.5 字典变小造成 hash collision 增加

如果从 `D_BITS=14` 降到 `13`：

- entry 数从 `16384` 降到 `8192`；
- 单 state 字典从 `64KB` 降到 `32KB`；
- hash collision 概率上升；
- 更容易覆盖有用候选；
- 压缩率可能下降，压缩吞吐也可能因 failed candidate 增加而下降。

因此不能只看显存下降。缩字典必须同时看：

- candidate hit rate；
- failed compare rate；
- match length 分布；
- 压缩率；
- kernel 和 no-ocl-init 总吞吐。

---

## 3. 当前 `lzo_gpu/hybrid` 实现现状

### 3.1 GPU 压缩

当前压缩路径：

- block 独立；
- 每个 active work-item 持有一份 dict slice；
- 同一 work-item 可轮转处理多个 block；
- dict 使用 epoch 逻辑清空；
- 默认 `D_BITS=14`；
- 压缩 kernel 内部以 hash probe、candidate validation、match extension 和 token output 为主。

主要问题：

- 字典 state owner 与 work-item 绑定；
- block 多时字典池膨胀；
- slots cap 虽然能降低字典池，但也会降低并发和隐藏延迟；
- 压缩 kernel hot path 依赖当前 dict layout，不能先于字典设计随意改。

### 3.2 GPU 解压

当前解压路径：

- block 独立解压；
- 已有 token fast path / short copy 类改动；
- 已有 metadata 条件上传；
- Windows 上 chunked readback/write 更有价值，Linux 默认应保守。

解压的结构风险低于压缩，因为不需要维护 hash/dict 候选表。

### 3.3 Hybrid

`lzo_hybrid` 当前支持：

- GPU-only；
- OpenCL CPU-only；
- CPU+GPU mixed；
- `--gpu-ratio` 按 block range 切分；
- `--cpu-threads` 限制 OpenCL CPU slots；
- daemon 中复用 context、queue、program、kernel 和主要 buffer。

主要问题：

- adaptive 尚未成为真正模型；
- CPU/GPU 比例调度依赖稳定的 GPU-only 和 CPU-only 性能；
- 在核心字典/并发结构没有稳定前，不应优先调 hybrid 策略。

---

## 4. 后续改进阶段

### 4.1 P0：state-slot pool 与字典/并发解耦

目标：减少字典池膨胀，同时尽量保留 GPU 并发。

步骤：

1. **引入 state slot 概念**
   - `state_slot` 持有 dict；
   - work-item 负责执行；
   - 第一版可令 `global_size == state_slots`，先验证正确性。

2. **静态 range 映射**
   - 每个 state slot 处理连续 block range；
   - 避免动态 task queue 的原子和同步；
   - 记录 `blocks/state`。

3. **state slot 预算**
   - 由 `nblk`、CU 数、dict bytes、显存预算共同决定；
   - 不再把 slots cap 当最终方案；
   - 第一轮只测少量 `CU * factor`，例如 `4/8/12/16`，避免无边界扫描。

4. **验证**
   - block size：`32/48/64KB`；
   - `D_BITS=14` 固定；
   - 指标：`CompKernel`、no-ocl-init `CompTotal`、压缩率、dict bytes、state slots、blocks/state；
   - 全样本 roundtrip。

可靠性：**中高**。它直接处理根因，但收益取决于调度是否能维持 occupancy。

### 4.2 P1：D_BITS 与 state-slot 联动

目标：确认是否有必要减小单 state 字典。

步骤：

1. 在 P0 成立后扫描 `D_BITS=13/14`；
2. 暂不优先 `D_BITS=15`，因为资源压力过大；
3. 统计压缩率损失与吞吐收益；
4. 增加 debug counter：
   - valid candidate 数；
   - failed candidate 数；
   - match length 分布；
   - boundary/cold-start 估计。

判定：

- 如果 `D_BITS=13` 压缩率损失约 `<=1%` 且 kernel/total 明显提升，可作为部分配置候选；
- 如果吞吐提升不稳定或压缩率损失集中在文本/内存镜像等关键样本，保留 `D_BITS=14`。

可靠性：**中**。缩字典一定降低资源，但压缩率和候选质量风险明确。

### 4.3 P2：适配新字典结构的压缩 kernel hot path

目标：在 state-slot 和 D_BITS 策略稳定后，再优化 probe 热路径。

步骤：

1. **dict entry 单次加载**
   - 尽量一次 load 取 epoch/offset；
   - 避免增加额外签名字段。

2. **候选验证前移**
   - 先做范围、epoch、最小距离判断；
   - 失败即返回 literal path。

3. **短匹配固定路径**
   - 3/4 字节初筛固定化；
   - match extension 只处理通过初筛的候选。

4. **输出路径整理**
   - literal 累积轻量化；
   - match token 输出只在确认 match 后进入重路径。

验证：

- 必须证明 `dict_load / candidate_load / failed_compare / extension_steps` 至少一项下降；
- 不接受结果分歧后无脑 gate；
- 不再做 `sig4/sig8`、bench-only gather/compaction、多 kernel 拆分压缩。

可靠性：**中**。方向正确，但必须建立在 P0/P1 后。

### 4.4 P3：解压优化

步骤：

1. token decode fast path；
2. short match copy；
3. 安全距离下 4/8 字节 wide copy；
4. chunked readback/write 平台化默认；
5. metadata 条件上传保持。

可靠性：**中高**。解压路径边界清楚，但 wide copy 必须严格验证重叠 match。

### 4.5 P4：Hybrid 调度

步骤：

1. 固定 GPU-only 和 OpenCL CPU-only 基线；
2. 扫描 `gpu_ratio` 与 `cpu_threads`；
3. 建立离线分段表；
4. daemon 中用最近窗口吞吐做在线修正；
5. 小文件避免 mixed，大文件才启用 mixed。

可靠性：**中**。必须等 GPU 核心路径稳定后再推进。

### 4.6 P5：Daemon / 边缘信息流验证

步骤：

1. `raw-buffer-session` 长连接；
2. byte/time window batch；
3. 输出 queue/build/codec/readback/publish 分段；
4. structured-stream 中比较 `none/lz4/lzo_cpu/lz4_gpu/lzo_gpu`。

可靠性：**系统层中高、核心压缩低**。只用于端到端验证，不证明核心 kernel 更优。

---

## 5. 论文表述建议

`lzo_gpu/hybrid` 不应写成“提出新 LZO 算法”。更准确的表述是：

> 面向大块状态压缩与异构设备执行的 OpenCL LZO 框架，通过字典 state 预算、块级并行、可复用 GPU runtime、解压输出排水优化和 CPU/GPU range 调度，把传统 LZO 分块压缩扩展成可在真实系统路径中进一步验证净收益的异构压缩后端。

## 6. 当前最终对比

最终对比结果见 `exp_results/remote105_gpu_compare_full6`。该结果来自 `192.168.2.105` 的 `/root/samples` 全量 26 文件，配置为 `lzo1x / D_BITS=14,15 / 48KB,64KB / local=1`，每文件 `1` 轮 bench + `6` 轮真实压缩/解压。

相对 `lzo_gpu_baseline_`：

- 压缩率：全部配置中位差异 `0.00%`，`clear16` 不改变候选表达能力或输出格式；
- 压缩 kernel：中位 `-1.05%`，平均 `-0.73%`，105 上没有形成核心压缩内核收益；
- 解压 kernel：中位 `+0.09%`，平均 `+0.07%`，基本持平；
- 端到端压缩：中位 `+4.25%`，平均 `+5.51%`，主要来自字典池缩小和真实路径开销下降；
- 端到端解压：中位 `-0.31%`，平均 `-0.25%`，基本持平。

当前已迁移到 `lzo_gpu` 的有效项：`clear16` 字典、按 block size 选择压缩 program、daemon 按 `alg × D_BITS × dict_mode` 缓存、解压 token/short-copy fast path、daemon raw-buffer 协议与 context/program/kernel 复用。未迁移项：sig4/sig8、bench-only gather/compaction、多 kernel 拆分压缩、文件特征 gate、未验证的更宽 compare/copy。

提交前重新收集的 105 当前 GPU 基线见 `exp_results/gpu_baseline_105_final`，用于后续改进轮次作为新基线。

