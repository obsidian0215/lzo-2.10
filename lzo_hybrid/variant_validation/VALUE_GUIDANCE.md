# LZO Hybrid 变体测试价值指引

更新时间：2026-04-16

## 1. 高测试价值方向

- 先把当前 CPU/GPU 任务切分、fallback、同步路径建账；
- 用 7~9 次 manual roundtrip 建立分段时间基线；
- 把无稳定收益却增加状态空间的 host/scheduler 开关优先冻结或删除。

## 2. 低测试价值方向

- 同时改 `D_BITS`、算法、调度模型；
- 只看 bench，不看 manual 分段时间；
- 在没有建立 baseline inventory 的前提下直接堆 scheduler 启发式。
