# LZO GPU 变体测试价值指引

更新时间：2026-04-16

## 1. 高测试价值方向

### 1.1 先把默认源码收敛成单 hash、低分支基线

优先级最高的是：

- 把默认内核里的 `主备 hash`、fingerprint、额外 copy-match 分支从默认路径里剥出来；
- 先确认“更简单的基线”是否更稳，再让附加机制以候选身份重新进入对比。

### 1.2 单阶段、单机制的 dictionary / copy-match 对比

高价值候选包括：

- `hash_dict / primary_secondary_hash`
- `hash_dict / fingerprint_filter`
- `kernel_dec / match_copy / copy_match_branch_prune`
- `kernel_dec / match_copy / vector_width`

这类条目的归因最清楚，也最接近当前源码清理目标。

### 1.3 与 `D_BITS=13/14/15` 分离验证

`D_BITS` 本身会改变命中密度与 dictionary 压力，因此：

- hash 机制验证要固定 `D_BITS`；
- `D_BITS` 扫描要与 hash/fp/copy-match 分开做。

## 2. 低测试价值方向

### 2.1 一次同时改 `D_BITS + hash + copy-match`

这会让结果无法归因，直接拉低记录价值。

### 2.2 保留默认源码中的重分支解压路径，再去比较 hash

如果默认 `COPY_MATCH()` 已经带着额外分支深度，先测 hash 的意义会被噪声污染。

### 2.3 重新扩散 `999` 路线

当前主线明确不做 `999` 扫描；任何把 `999` 混回主线的尝试都属于低优先级。

## 3. 当前经验结论

- `主备 hash` 应从默认基线中剥离，作为 **待验证候选**；
- `fingerprint` 对 LZO 目前也应视作 **候选机制**，而不是默认前提；
- `COPY_MATCH()` 里只增分支深度、没有额外向量收益的小 offset 特化，应优先收敛；
- 主机端与 `hybrid` 侧同样要先建账，再决定保留哪些功能。

## 4. 立项前自检

每个新候选在进入正式验证前，至少回答：

1. 它属于哪个 `stage / operation`？
2. 它是否只是把默认源码里的附加机制重新拿出来当候选？
3. 它是否与 `D_BITS`、host、scheduler 变化充分隔离？
4. 若失败，最可能坏在哪个主判据？
