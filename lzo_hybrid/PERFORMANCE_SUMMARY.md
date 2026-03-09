# LZO Hybrid 性能总结

更新时间：2026-03-09  
程序路径：`/root/lzo-2.10/lzo_hybrid/lzo_hybrid`

## 1. 当前正式结果工件

- stitched full-corpus artifact：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/lzo_param_sweep_merged.csv`
- provenance manifest：`/root/lzo-2.10/exp_results/runs/20260309_merged_full_83/merge_manifest.json`
- fresh analysis：`/root/analysis/20260309_full_refresh/lzo_best_per_file_medians.json`

这里的 hybrid 结论全部来自同一个 83-file matched corpus，不再混用旧 36-file baseline 或 subset-only 结果。

## 2. 83-file raw medians（lean matrix）

| Mode | Verified rows | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| lzo1x fixed | 249 | 748.03 | 745.72 | 2403.72 | 2379.64 | 21.63 | 12.29 |
| lzo1x adaptive | 249 | **906.62** | **833.92** | **2432.70** | 2221.36 | 21.63 | 12.27 |
| lzo1y fixed | 249 | 748.44 | 742.69 | 2340.33 | **2353.95** | 21.73 | 12.27 |
| lzo1y adaptive | 249 | **913.59** | **833.89** | **2411.78** | 2217.39 | 21.93 | 12.28 |

## 3. 83-file best-per-file medians

| Mode | Verified files | Comp total MB/s | Dec total MB/s | Comp kernel MB/s | Dec kernel MB/s | Ratio % | Comp power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| lzo1x fixed | 83 | 925.46 | 821.24 | 2477.15 | 2292.82 | 21.68 | 12.28 |
| lzo1x adaptive | 83 | 927.49 | 897.38 | 2580.50 | 2748.89 | 21.68 | 12.27 |
| lzo1y fixed | 83 | 926.01 | 812.25 | 2477.46 | 2256.97 | 22.12 | 12.26 |
| lzo1y adaptive | 83 | **942.86** | **900.74** | **2643.72** | 2673.32 | 22.12 | 12.27 |

对照 standalone best-per-file medians：

- `GPU lzo1x`：`1334.63 / 808.86`
- `GPU lzo1y`：`1330.73 / 808.09`

## 4. winner counts（83-file matched corpus）

### lzo1x

| Metric | GPU | Hybrid fixed | Hybrid adaptive |
|---|---:|---:|---:|
| Compression total | **62** | 17 | 4 |
| Decompression total | 23 | 12 | **48** |

### lzo1y

| Metric | GPU | Hybrid fixed | Hybrid adaptive |
|---|---:|---:|---:|
| Compression total | **60** | 18 | 5 |
| Decompression total | 23 | 8 | **52** |

## 5. 结果分析

### 5.1 LZO hybrid 的最新形态是“压缩看 GPU，解压看 adaptive”

这次结果最重要的新现象不是 hybrid 全面反超，而是角色分化：

- compression：GPU 明显更强，winner count 和 medians 都领先
- decompression：adaptive hybrid 在 `lzo1x` / `lzo1y` 上都拿到了更高的 best-per-file medians，并赢下更多文件

### 5.2 Adaptive 不再弱于 fixed

与更早版本相反，这轮 fresh 83-file stitched artifact 里 adaptive 在 LZO 上已经系统性强于 fixed：

- raw medians：adaptive comp/dec 都高于 fixed
- best-per-file medians：adaptive comp/dec 都高于 fixed
- file winner counts：adaptive 在 dec 上优势非常明显

因此当前文档不能再写“LZO fixed 明显强于 adaptive”。那已经被 fresh artifacts 推翻了。

### 5.3 但这不等于 hybrid 取代 GPU

即便 adaptive hybrid 在 decompression 上很强，它的 compression medians 仍显著低于 GPU-only：

- `GPU lzo1x` compression total：`1334.63`
- `Adaptive lzo1x` compression total：`927.49`
- `GPU lzo1y` compression total：`1330.73`
- `Adaptive lzo1y` compression total：`942.86`

所以当前正确表述是：**LZO hybrid adaptive 是很强的解压协同模式，但不是新的全局默认 engine。**

## 6. 跨算法位置（matched 83-file corpus）

来自 `cross_family_best_per_file_medians.csv`：

- `LZ4 Hybrid fixed` compression total `1425.90 MB/s`，高于全部 `LZO Hybrid` 模式
- `LZO Hybrid adaptive` decompression total `897.38 ~ 900.74 MB/s`，高于 `LZ4 Hybrid fixed/adaptive`

因此当前 cross-family 结论应写成：

- hybrid compression：LZ4 更强
- hybrid decompression：LZO adaptive 更强

## 7. 当前结论

> `lzo_hybrid` 的 fresh 83-file stitched results 已经不再支持“fixed 优于 adaptive”的旧结论。当前最准确的描述是：GPU 仍是 LZO compression 的默认主路径，而 adaptive hybrid 在 decompression 上已经形成了明显优势，是当前 LZO family 中最值得继续优化的协同模式。
