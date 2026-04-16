# LZO GPU 变体验证标准

更新时间：2026-04-16

## 1. 目标

本标准用于约束 `lzo_gpu/variant_validation/` 下所有 NVIDIA / Intel 平台的变体验证工作，目标是让：

- 目录结构统一；
- `lzo1x` / `lzo1y` / `D_BITS` / 过程机制的验证口径一致；
- Intel/Linux 自动验证未来可以直接复用同一套说明；
- 压缩率与吞吐量的 trade-off 判定不再靠口头解释。

## 2. 目录契约

```text
lzo_gpu/variant_validation/
  README.md
  VALIDATION_STANDARD.md
  PROCESS_STAGE_CATALOG.md
  VALUE_GUIDANCE.md
  VARIANT_RECORD_TEMPLATE.md
  KERNEL_VARIANTS.md
  HOST_VARIANTS.md
  nvidia/
    variants/
    scripts/
    results/
    logs/
    work/
  intel/
    variants/
    scripts/
    results/
    logs/
    work/
```

约束如下：

- 根目录只放标准文档；
- `nvidia/`、`intel/` 放平台相关的本地工作区；
- 新增的变体实现、脚本、结果，默认只允许进入对应 vendor 目录；
- 自动化脚本必须依赖这套目录语义，而不是依赖历史散落路径。

### 2.1 根层组件记录文档契约

`lzo_gpu` 根目录必须同时维护：

- `KERNEL_VARIANTS.md`
- `HOST_VARIANTS.md`

要求如下：

- 组件记录按 `vendor -> optimization_object -> stage -> operation` 组织；
- 每个条目至少覆盖：变体名称、动机、设计与实现、整体结果、判定、证据路径；
- 新候选进入 full / recheck 前，必须先在根层组件账本中占位；
- 正式结果落地后，必须同时回写 `summary.md`、`records/...` 与根层组件账本。

## 3. 样本与环境契约

### 3.1 样本根目录

样本必须来自统一样本仓：

- 当前 Windows/NVIDIA 默认：`C:\Users\Administrator\Documents\git-repo\samples`
- Intel/Linux 自动验证默认：`/root/samples`
- 若 Intel/Linux 环境覆盖该路径，仍必须映射到相同内容的数据集。

统一使用：

- `SAMPLES_ROOT`（Intel/Linux 默认值为 `/root/samples`）

### 3.2 环境变量

脚本应显式记录并回放：

- `SAMPLES_ROOT`
- `FORCE_OPENCL_DEVICE`（如使用）
- `LZO_GPU_DIR`
- `LZO_GPU_NO_CLBIN`（如使用）
- 其他影响 bench / standalone 结果的环境变量

### 3.3 执行方式

- 所有正式测试都必须前台阻塞执行；
- 候选只与上一迭代稳定版本做配对比较；
- 验证阶段禁止临时切换 build 工具链；
- 采用“预先准备好的变体目录 + 手动指定内核/工件”方式完成 A/B。

补充约束：

- `smoke` 只负责验证**内核可加载 / roundtrip 正确 / 指标可解析**；
- `smoke` 通过后默认直接删除中间结果，不再保留 `results/*smoke*` 作为正式结论；
- 进入报告与排序的，只能是 full / recheck / gate 级结果。

### 3.4 正式固定参数

除非某一轴被明确声明为“正在单独扫描的正式变体”，否则 `lzo_gpu` 的正式 full / recheck / gate 结果必须锁定：

- `block = 64KB`
- `localsize = 1`
- `level = 14`

补充要求：

- `algo` 与 `D_BITS` 若不是本轮正式扫描轴，就必须在同一对比里保持固定；
- `bench` 与 `manual roundtrip` 必须显式传入相同的 `block/local/level`，禁止再依赖二进制默认值；
- `run_config.json`、`summary_comparison.*`、manifest 与记录文档都必须回显这组固定参数。

### 3.5 组件自动化入口

`lzo_gpu` 的正式自动化入口按组件拆分，至少应维护：

- `nvidia/scripts/kernel_comp/validate_lzo_kernel_comp_stable.ps1`
- `nvidia/scripts/kernel_dec/validate_lzo_kernel_dec_stable.ps1`
- `nvidia/scripts/host/validate_lzo_host_stable.ps1`
- `nvidia/scripts/shared/` 下的共享驱动

对应结果目录也必须按组件落到：`results/kernel_comp/`、`results/kernel_dec/`、`results/host/`。

每次正式运行至少要产出：

- `run_config.json`
- `raw_runs.csv`
- `sample_medians.csv`
- `candidate_comparison.csv`
- `control_comparison.csv`
- `summary_by_role.csv`
- `summary_comparison.csv`
- `summary_comparison.txt`

## 4. 变体包标准

每个变体建议放在：

```text
<nvidia|intel>/variants/<variant_id>/
```

每个变体目录建议至少包含：

- 对应算法/级别的内核文件（如 `lzo1x*.cl`、`lzo1y*.cl`、`*.clbin`）；
- 二进制或其定位说明；
- `variant.json` 或 `variant.md`。

建议元信息字段：

- `id`
- `based_on`
- `algo`
- `dbits`
- `stage`（如 `hash_dict`、`matchcopy`、`count_loop`）
- `device_vendor`
- `compile_mode`
- `notes`

对于 `host` 组件，变体目录还必须补充**主机程序/源码记录包**，至少覆盖：

- 当前用于正式验证的可执行程序、启动脚本或运行时目录；
- 关联的主机侧源码文件列表；
- 构建说明、环境变量开关、补丁摘要；
- 若只是环境开关差异，也必须把实际开关值写进 manifest 与结果总结。

补充要求：

- 默认源码应优先保持**最简稳定基线**；
- fingerprint、主备 hash、branch-heavy copy-match 等附加机制，若收益未证实，应先以独立候选进入 `variants/` 与根层组件账本，而不是常驻默认路径。

### 4.1 特殊轴：`hash_table_overhead`

`hash_table_overhead` 是一个**跨 kernel + host 的特殊正式轴**，用于记录：

- kernel 侧 dictionary / slot 数量 / entry 布局 / 管理逻辑带来的命中与吞吐影响；
- host 侧对应的 buffer alloc / resize / staging / upload/download 开销与显存占用影响。

凡是触碰该轴的候选，必须同时：

- 在 `KERNEL_VARIANTS.md` 与 `HOST_VARIANTS.md` 建同名或可一一映射的条目；
- 在 manifest 与明细记录中写明 `special_axis_tags = ["hash_table_overhead"]`；
- 同时给出 kernel 指标与 host 分段时间，不能只报其中一边。

## 5. 标准验证流程

### 5.1 准备

1. 选定稳定基线与候选；
2. 固定算法、`D_BITS`、块大小、bench 时长；
3. 在 vendor 目录下准备 runtime-ready 变体包；
4. 记录 commit、设备、驱动、环境变量快照。

### 5.2 执行

每个候选至少产出两类结果：

1. **Track-K**：`--bench` 为主判；
2. **Standalone**：`-v` roundtrip 作为工程映射验证。

### 5.2.1 稳定性协议（正式结果强制执行）

正式 full / recheck / gate 结果必须走稳定性协议。

内核改动默认执行序列：

1. `control_a`（基线）
2. `candidate`
3. `control_b`（同一基线）

要求如下：

- 固定样本顺序、参数、环境变量；
- 不再单独拆“守门阶段”；
- 每个样本、每个角色统一执行：
  - **`2` 次 `--bench`**；
  - **`5` 次手动 standalone roundtrip**；
- bench 指标按 2 次结果取中位数；manual 指标按 5 次结果取中位数；
- 必须同时产出：
  - `candidate vs baseline_ref`；
  - `control_b vs control_a`（control 噪声包络）。

对于 `host` 组件，补充要求：

- 正式结果主协议改为 **7~9 次 manual roundtrip**；
- 必须记录分段时间、总时间、ratio；
- `bench` 只允许做 smoke，不得作为主机侧主排序依据。
- 当前正式 host 变体范围只包含：`standard_copy / mapped`、`pipeline`（含 overlap / steady-state submit path）、`pack / compaction gate`；
- `metadata / header` 修补、`telemetry`、bench-only 采样与日志开关，默认只记为**修正 / hygiene**，不作为正式 host 变体 ID 建账。

判读要求：

- 候选收益若未明显越过 control-vs-control 漂移，不得直接判为 `adopt`；
- 不能只报 candidate 平均值，不报 control 漂移。

### 5.3 归档

结果目录规范：

```text
<nvidia|intel>/results/<stage>_<variant>_<yyyymmdd>/
```

最少包含：

- `raw_results.csv`
- `summary_by_variant.csv`
- `summary_comparison.csv`
- `summary.md`
- 原始日志

若采用稳定性协议，建议额外包含：

- `raw_runs.csv`
- `sample_medians.csv`
- `control_comparison.csv`
- `candidate_comparison.csv`

## 6. 判定口径

### 6.1 指标方向

必须明确写在报告里：

- `Bench Comp` / `Bench Dec` / `Comp Kernel` / `Dec Kernel`：**越高越好**
- `Comp Total` / `Dec Total`：**越低越好**
- `ratio = compressed_size / original_size`：**越低越好**

### 6.1.1 内核改动与主机改动的判读分离

必须先判断本轮候选属于哪一类：

- **内核改动**：hash/dict、matchcopy、count loop、unaligned/vector copy 等内核路径改动；
- **主机改动**：`standard_copy / mapped`、pipeline、pack/compaction、queue/event 等 steady-state host/runtime 路径改动。

补充说明：

- `metadata / header` 修补、`telemetry` 与 bench-only 逻辑，若不改变 steady-state 数据路径，默认归为**修正 / hygiene**，不进入 `HOST_VARIANTS.md` 的正式候选池；
- 真正需要主协议验证的 host 候选，应优先回答：它是否改变了 steady-state 的数据搬运、提交重叠或 pack/compaction 行为。

判读规则：

- **内核改动**：主要看 `Bench Comp/Bench Dec` 与 `Comp Kernel/Dec Kernel`，`Total` 只作补充参考；
- **主机改动**：主要看分段时间，不能直接拿包含 `OCI Setup/OCL init` 的总吞吐做主判。

即：

> 内核改动，先看 kernel throughput 和 ratio；主机改动，先看分段时间并剔除 `OCI Setup/OCL init` 这类不稳定大头。

### 6.2 必报字段

每轮至少输出：

- `avg`
- `pos`
- `neg`
- `neg_worst`

如存在明显 trade-off，建议补充：

- `pos_best`
- 代表性样本
- `tradeoff_note`

### 6.3 当压缩率与吞吐量方向相反时

这是强制分析项。

#### 情形 A：压缩率改善（`ratio` 下降），吞吐略退

可接受，但必须满足：

- 吞吐退化幅度小且可控；
- `neg_worst` 没有越过当前阶段门限；
- 代表样本不出现系统性崩塌；
- 报告中明确写出“用多少吞吐换来了多少压缩率改善”。

默认倾向：

> **如果压缩率更好，而吞吐只出现较小回退，可以作为可接受候选继续推进。**

#### 情形 B：吞吐改善，压缩率变差（`ratio` 上升）

不得自动判优，必须回答：

1. 吞吐收益是否明显大于压缩率回退？
2. 回退是否只在局部样本出现？
3. 最差样本是否可接受？
4. 当前阶段目标是否允许为吞吐牺牲压缩率？

只有答案整体为正面时，才允许进入 `watch` 或 `adopt`。

### 6.3.1 对 `Total` 指标的使用约束

- 对**内核改动**，`Comp Total/Dec Total` 只用于辅助发现异常尾部，**不作为主排序依据**；
- 对**主机改动**，若要看总吞吐，必须同时给出分段时间，并显式说明是否排除了 `OCI Setup/OCL init` 这类不稳定大头；
- 没有分段时间支撑时，不得只凭 `Total` 给主机改动下结论。

### 6.4 判定标签

建议统一使用：

- `adopt`
- `watch`
- `reject`

### 6.5 阶段组合策略（固定锚点 + 延后组合）

LZO 内核阶段优化统一采用：

> **固定锚点基线 + 单阶段筛选 + 延后组合扫描**

规则如下：

1. 同一轮迭代里，所有单阶段候选都先只对比同一个稳定锚点基线；
2. 不采用“某阶段一通过就立刻叠加成下一基线”的滚动方式；
3. 每个阶段先筛出 `watch`/`adopt` 候选，再做组合扫描；
4. 组合扫描统一对比同一锚点基线，避免跨阶段结果失真；
5. 组合优胜者通过稳定性复核后，才允许升级为下一轮全局基线。

## 7. Intel/Linux 自动化兼容要求

Intel 自动验证脚本必须满足：

- 使用与 NVIDIA 相同的目录语义；
- 默认使用 `SAMPLES_ROOT=/root/samples`，并允许显式覆盖；
- 通过 manifest 识别算法、级别、变体，而不是依赖手工记忆；
- 输出同名核心工件（`raw_results.csv`、`summary_comparison.csv`、`summary.md`）。

Intel 侧详细执行契约见：`INTEL_AUTORUNNER_CONTRACT.md`。

## 8. 报告最小模板

1. 变体信息：`baseline -> candidate`
2. 样本、算法、级别、参数、设备、环境变量
3. 正确性：roundtrip / verify
4. 主指标：`Bench Comp/Bench Dec`、`Comp Kernel/Dec Kernel`、`ratio`
5. 稳定性：`2 bench + 5 manual` 轮数、中位数口径、control-vs-control 漂移
6. 配对统计：`avg/pos/neg/neg_worst`
7. trade-off 结论：收益是否覆盖代价
8. 最终标签：`adopt/watch/reject`
9. 根层组件账本回写位置：`KERNEL_VARIANTS.md` 或 `HOST_VARIANTS.md`
