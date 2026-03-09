# LZO GPU 性能总结

更新时间：2026-03-09  
程序路径：`/root/lzo-2.10/lzo_gpu/lzo_gpu`

## 1. 当前正式结果工件

- stitched full-corpus artifact：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/lzo_param_sweep_merged.csv`
- provenance manifest：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/merge_manifest.json`
- 基础 70-file run：`/root/lzo-2.10/exp_results/runs/20260309_201631/`
- single-file completion runs：`/root/lzo-2.10/exp_results/runs/20260309_222029/` ~ `20260309_224304/`
- 跨算法汇总：`/root/analysis/20260309_full_refresh/`

说明：LZO 这次正式工件是一个 **lossless stitched artifact**。它由一个 70-file base run 和 14 个 single-file completion runs 组合而成，83 files 全部来自同一个 `/root/samples` corpus，并通过 manifest 记录了每个文件来自哪个 run。

## 2. 测试口径

- 语料：`/root/samples`，83 files
- 只使用 `Roundtrip_OK=yes` rows
- `lzo1x` 和 `lzo1y` 分开统计，不把二者揉成一个“LZO family 单数值”
- 跨算法比较只在 matched 83-file corpus 上成立

## 3. 83-file best-per-file medians

| Engine | Verified files | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU lzo1x | 83 | 615.68 | 642.44 | 2074.46 | 1765.48 | 21.68 | 14.74 |
| CPU lzo1y | 83 | 613.45 | 642.66 | 1994.41 | 1756.91 | 22.12 | 14.80 |
| GPU lzo1x | 83 | **1334.63** | **808.86** | **5828.83** | **11600.56** | **23.38** | **13.85** |
| GPU lzo1y | 83 | 1330.73 | 808.09 | 5643.52 | 11587.72 | 23.19 | 14.15 |

对应 raw medians（用于看整张 lean matrix 的整体分布）：

| Engine | Comp total MB/s | Dec total MB/s |
|---|---:|---:|
| GPU lzo1x | 756.26 | 774.24 |
| GPU lzo1y | 753.57 | 766.11 |

## 4. 结果分析

### 4.1 GPU 仍是 LZO compression 的默认主路径

无论是 `lzo1x` 还是 `lzo1y`，GPU best-per-file compression median 都在 `1330+ MB/s`，明显高于 CPU 的 `613~616 MB/s`。这说明在当前 matched 83-file corpus 上，LZO GPU 仍然是默认 compression leader。

### 4.2 两条算法线几乎并列

`lzo1x` 与 `lzo1y` 在当前 GPU 结果上的差距很小：

- compression total：`1334.63` vs `1330.73`
- decompression total：`808.86` vs `808.09`

因此当前更重要的不是 `1x` / `1y` 谁“绝对胜出”，而是它们都形成了稳定的 GPU 高吞吐路径。

### 4.3 kernel/headroom 仍明显高于 delivered total

以 `GPU lzo1x` 为例：

- comp kernel `5828.83 MB/s` vs comp total `1334.63 MB/s`
- dec kernel `11600.56 MB/s` vs dec total `808.86 MB/s`

这和 LZ4 一样说明：runtime / file path / container handling 仍然是系统结论的重要限制项。

## 5. 跨算法位置（matched 83-file corpus）

来自 `cross_family_best_per_file_medians.csv`：

- `LZ4 GPU` comp/dec totals 为 `1497.76 / 1085.39`
- `LZO GPU lzo1x` 为 `1334.63 / 808.86`
- `LZO GPU lzo1y` 为 `1330.73 / 808.09`

因此当前最稳妥的 cross-family 结论是：**在这个 matched corpus 和当前参数矩阵下，LZ4 GPU 是更快的 standalone 路径；LZO GPU 仍然很强，但略低于 LZ4 GPU。**

## 6. 当前结论

> `lzo_gpu` 已经通过 stitched 83-file artifact 形成了新的正式 baseline。无论是 `lzo1x` 还是 `lzo1y`，GPU 都显著领先 CPU，并且 `lzo1x` / `lzo1y` 之间差距很小；在当前 matched cross-family 结果里，LZO GPU 位于 LZ4 GPU 之后、但仍然是 LZO 家族最强的 standalone compression baseline。
