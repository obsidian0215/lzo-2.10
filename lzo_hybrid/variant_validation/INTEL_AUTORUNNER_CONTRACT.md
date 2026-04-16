# LZO Hybrid Intel/Linux 自动 Runner 契约

更新时间：2026-04-16

## 1. 固定前提

- 操作系统：Linux
- 样本根目录：`/root/samples`
- Runner 必须默认：`SAMPLES_ROOT=/root/samples`
- `hybrid` 正式主线以 **manual roundtrip** 为主，不以 `bench` 为主判据。

## 2. 必需输入参数

- `--baseline <variant_id>`
- `--candidate <variant_id>`
- `--component <host|scheduler>`
- `--algo <lzo1x|lzo1y>`（如适用）
- `--dbits <13|14|15>`（如适用）
- `--rounds <7|9>`
- `--sample-list <file>` 或 `--sample-glob <glob>`

## 3. 输出

Runner 必须生成：

- `manual_runs.csv`
- `segment_breakdown.csv`
- `summary_comparison.csv`
- `summary.md`

## 4. 判定要求

- 必报 ratio 变化
- 必报 `avg/pos/neg/neg_worst`
- 必报 trade-off 结论
- 不得在缺少分段时间时只看总时间给通过结论
