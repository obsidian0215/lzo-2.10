# LZO GPU 变体验证目录

更新时间：2026-04-16

## 1. 目录用途

这里是 `lzo_gpu` 的平台分层变体验证工作区，用来统一：

- `D_BITS=13/14/15` 扫描；
- `lzo1x` / `lzo1y` 对照；
- `hash_alt` / 双索引类改动；
- `COPY_MATCH`、向量复制、长度探测等过程机制验证。

## 2. 目录结构

```text
lzo_gpu/variant_validation/
  README.md
  VALIDATION_STANDARD.md
  PROCESS_STAGE_CATALOG.md
  VALUE_GUIDANCE.md
  VARIANT_RECORD_TEMPLATE.md
  INTEL_AUTORUNNER_CONTRACT.md
  variant_manifest.template.json
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

说明：

- 根层只放标准文档；
- `nvidia/` 用于当前 Windows/NVIDIA 验证；
- `intel/` 预留给 Intel/Linux 自动验证；
- 新的 LZO 变体实现、验证脚本、结果目录后续一律优先进入对应 vendor 子目录。
- `KERNEL_VARIANTS.md` / `HOST_VARIANTS.md` 用来做根层组件建账，按 `vendor -> stage -> operation` 记住每个候选的动机、设计、结果与判定。

Intel/Linux 自动验证固定样本根目录：`/root/samples`。

## 3. 当前迁移策略

从现在开始：

- **LZO 新增变体实现** → `lzo_gpu/variant_validation/nvidia/variants/`
- **LZO 新增验证脚本** → `lzo_gpu/variant_validation/nvidia/scripts/`
- **LZO 新增结果工件** → `lzo_gpu/variant_validation/nvidia/results/`

历史根部目录中的 `tools/`、`exp_results/` 仍保留作为过渡期入口，但不再继续扩散新资产。

补充：

- 当前默认 `lzo1x.cl` / `lzo1y.cl` 已收敛到**单 primary hash 基线**；
- `主备 hash / secondary hash probe` 与 fingerprint 统一视作**待验证变体**，不再常驻默认路径；
- `COPY_MATCH()` 中增加分支深度、又没有额外向量收益的小 offset 分支已先做源码收敛，后续再单独验证是否值得保留。

## 4. 当前 LZO 验证范围

当前计划中的 LZO 主线范围为：

- `lzo1x` / `lzo1y`
- `D_BITS = 14 / 15`
- `hash_alt` / `DINDEX_ALT`
- `COPY_MATCH` / 匹配复制相关过程路径
- 其他局部过程机制（如向量复制宽度、长度统计回路）

明确排除：

- 当前阶段**不做 `999` 扫描**。

## 5. 结果判定提醒

> 口径提醒：`ratio = compressed_size / original_size`，因此 **越低越好**。

这意味着：

- 如果 `ratio` 下降而吞吐只小幅回退，可以接受并继续推进；
- 如果吞吐上升但 `ratio` 上升，则代表压缩率更差，必须做 trade-off 分析，不能直接判优。

具体判定规则以 `VALIDATION_STANDARD.md` 为准。

## 6. Intel 自动验证约束

为了后续 Intel/Linux 自动验证，当前文档与目录约束要求：

1. 通过 `SAMPLES_ROOT` 注入样本根目录；
2. 通过 manifest 标明算法、级别、阶段、变体来源；
3. 结果产物统一命名；
4. 不把 Windows 绝对路径写死到标准流程里。

Intel 自动 runner 约束与 manifest 约定见：`INTEL_AUTORUNNER_CONTRACT.md`、`variant_manifest.template.json`。

## 7. 组件记录要求

- `KERNEL_VARIANTS.md`：登记内核组件（`kernel_comp` / `kernel_dec`）
- `HOST_VARIANTS.md`：登记主机组件（`host`）

每条记录至少写：

- 具体名称
- 动机
- 设计思路和实现
- 整体结果
- 判定
- 证据路径

## 8. 后续动作建议

1. 先把 NVIDIA 侧的新变体验证资产收束到本目录；
2. 再补 Intel/Linux 自动 runner，使其消费同一套目录和文档；
3. 后续每个 LZO 阶段结果都必须输出：
   - 吞吐变化
   - ratio 变化
   - `avg/pos/neg/neg_worst`
   - trade-off 结论
