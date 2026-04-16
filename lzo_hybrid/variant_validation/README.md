# LZO Hybrid 变体验证目录

更新时间：2026-04-16

## 1. 目录用途

这里是 `lzo_hybrid` 的**主机端 / 任务调度** 变体验证工作区。

本目录不重复记录 `lzo_gpu` 的内核候选，而是专门跟踪：

1. `host`：buffer、上传下载、文件与 metadata、同步与回退；
2. `scheduler`：CPU/GPU 分工、chunk 分配、fallback threshold；
3. `lzo1x` / `lzo1y` / `D_BITS` 在混合执行下的调度差异；
4. Intel/Linux 与 Windows/NVIDIA 的同名变体对照。

## 2. 根层文档

```text
lzo_hybrid/variant_validation/
  README.md
  VALIDATION_STANDARD.md
  PROCESS_STAGE_CATALOG.md
  VALUE_GUIDANCE.md
  VARIANT_RECORD_TEMPLATE.md
  INTEL_AUTORUNNER_CONTRACT.md
  variant_manifest.template.json
  HOST_VARIANTS.md
  SCHEDULER_VARIANTS.md
  nvidia/
  intel/
```

## 3. 记录原则

- 只维护 `HOST_VARIANTS.md` 与 `SCHEDULER_VARIANTS.md` 两本账；
- 正式主判据必须来自 **7~9 次 manual roundtrip** 的分段时间；
- 若改动同时涉及内核与调度，必须拆到 `lzo_gpu` 与 `lzo_hybrid` 分别记录；
- 与 `D_BITS`、算法相关的信息应写入条目，但不允许与调度变化混成单个黑盒候选。
