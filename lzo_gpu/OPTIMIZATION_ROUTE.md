# LZO GPU 优化路线（可读版）

## 全集基线结果（当前保留）

- 基线全集目录：`/root/lzo-2.10/exp_results/runs/fullset_allcfg_current_lzo/runs/20260403_140710`
- 基线全集主结果：`lzo_param_sweep.csv`
- 主结果哈希：`sha256=ff8ad767f935253f43d7984e6fceb2c896c0ff1d53cf8d52d73d90764ddbe958`
- 配置汇总哈希：`sha256=be4a4a007a201d9e236ff2c7992d588e62fa02be538a27f86adf41c7fa6dd2ba`
- 行数与完整性：`rows=2300`，`Roundtrip_OK=2300/2300`，引擎覆盖 `CPU/GPU/HYBRID`
- GPU 引擎聚合：`CompTotal mean=1254.21 MB/s`，`DecTotal mean=2143.05 MB/s`，`Ratio mean=27.0331%`

## 已采纳修改

### 解压 metadata 条件上传（bench 去重上传）

动机

- 解压 bench 中 `off/lens` 元数据每轮重复上传，固定开销偏高，持续拖慢 `DecTotal`。

设计

- 增加 metadata 缓存与变化检测。
- 仅在 metadata 变化或 buffer 重建时上传 `off/lens`。
- kernel 等待事件列表按真实上传事件数动态传递，避免无效等待。

实现

- 修改文件：`/root/lzo-2.10/lzo_gpu/lzo_gpu.c`
- 关键实现点：`dec_meta_changed` 判定、`write_event_count` 动态化、上一轮 metadata 缓存数组复用。

测试结果

- 结果目录：`/root/lzo-2.10/exp_results/runs/gpu_dec_meta_cache_r1/results/`
- 结果文件：`lzo_subset_ab_cleanhead_v1.json`
- 基线工件：`.../artifacts/lzo_gpu_baseline_cleanhead`，`sha256=7f0d36be40211428cc8f3c8794b43ae3d974683cc92feb9a147a0d759d5ec854`
- 候选工件：`.../artifacts/lzo_gpu_candidate`，`sha256=20a9b1345dfa8b99b597188faeaf6d610bc56b440ed4c0145428cfaf4a85c517`
- 指标：`Comp -0.6415% / +0.1542%`，`Dec +0.6923% / +1.8585%`，`Ratio 0.0 pctpt`

采纳原因

- 解压均值/中位数稳定提升，压缩侧无门限外退化，保留到主实现。

## 未采纳修改（表格汇总）

| 修改名（实际语义） | 动机 | 设计与实现 | 测试结果 | 拒绝原因 |
| --- | --- | --- | --- | --- |
| 默认开启 pipeline | 期望提升大文件压缩吞吐 | 调整 pipeline 默认策略（证据：`/tmp/rebuild/lzo_gpu_iter1/`） | 运行失败：`no successful iteration` | 稳定性门禁失败 |
| 稀疏写回改 `writev` 聚合 | 降低写回系统调用开销 | host 稀疏写回聚合（证据：`reset_round_20260331_014206_LZG_RESET_R1B_subset_ab.json`） | `Comp -1.4895%/-0.0867%`，`Dec +0.1859%/+0.1537%` | 压缩侧负向 |
| non-pipeline compaction 打包回传 | 减少全量回传 | 增加 pack kernel 后下载 packed payload（证据：`..._R2_subset_ab.json`） | `Comp -3.7358%/-3.5477%`，`Dec +1.6381%/+0.4014%` | 压缩显著回退 |
| 去除压缩路径冗余同步 | 降低同步阻塞 | 去除两处额外 `clFinish/Flush`（证据：`..._R3_subset_ab.json`） | `Comp -0.1453%/-0.1076%`，`Dec +1.3677%/+0.1871%` | 压缩收益不足 |
| 自动 pipeline 门控（阈值+块数+熵门） | 让中大文件默认走重叠路径 | 引入 auto gate + entropy gate（证据：`/tmp/rebuild/lzo_gpu_iter4/`） | subset 解析失败 + 单文件复现失败 | 稳定性失败 |
| 保守窗口 pipeline 门控 | 修复激进门控副作用 | 仅 `192MB~512MB` 启用 pipeline（证据：`..._R5_subset_ab.json`） | `Comp -0.4507%/-0.5691%`，`Dec +1.5977%/-0.2864%` | 压缩门禁未过 |
| mapped 直写 + 非 compaction `writev` | 继续压缩排水阶段开销 | compaction 优先 mapped，非 compaction 稀疏写 `writev`（证据：`..._R6_subset_ab.json`） | `Comp -1.8806%/+0.1107%`，`Dec +0.5140%/+0.1350%` | 压缩门禁未过 |
| pipeline 排水非阻塞 + chunk 自适应 | 提升 pipeline 排水效率 | event wait 替代阻塞同步 + 自适应 chunk（证据：`..._R7_subset_ab.json`） | `Comp +0.2298%/+0.1670%`，`Dec +0.1868%/-0.1555%` | 压缩主判未过 |
| compaction 输出 mapped 直写优先 | 避免 map 成功后额外拷贝 | map 成功直写，失败回退 readback（证据：`..._R8_subset_ab.json`） | `Comp -0.0193%/+0.2748%`，`Dec +0.1754%/+0.0987%` | 压缩收益不足 |
| 调度/排水参数自适应（激进版） | 改善大 block 场景参数失配 | 并行度分段 + chunk 自适应 + pack local size 自适应（证据：`..._R9_subset_ab.json`） | `Comp -2.0295%/-0.2508%`，`Dec +0.3754%/-0.0520%` | 主判失败且最大回退过大 |
| 高 block 数保守降档 + 字典占用保护 | 缓解激进并行度回退 | 大 block 数场景下调 `wi_per_cu`（证据：`..._R10_subset_ab.json`） | `Comp +0.4097%/-0.1808%`，`Dec +0.0262%/+0.0288%` | 压缩中位数未过 |
| 仅 pipeline 低风险自适应 | 试探低风险收益 | chunk blocks + pack local size 自适应（证据：`..._R11_subset_ab.json`） | `Comp -0.3810%/+0.1828%`，`Dec +0.3519%/+0.0263%` | 压缩均值未过 |
| 保守组合（降档并行度 + pipeline chunk） | 组合有效片段争取净收益 | 组合 R10/R11 的保守项（证据：`..._R12_subset_ab.json`） | `Comp +1.0544%/-0.3900%`，`Dec +0.1089%/+0.0817%` | 压缩中位数未过 |
| null sink 下跳过 payload 回传与写盘 | 去掉 bench 固定 host 损耗 | `/dev/null` 场景仅保留必要统计（证据：`gpu_hostpath_deep_r1/runs/20260403_193713/lzo_subset_ab_coreonly_3s.json`） | `Comp -0.2698%/+0.5572%`，`Dec +0.7953%/+0.0419%` | 压缩主判未跨门限 |

## 当前代码一致性检查结论

- `lzo_gpu.c` 仍保留“解压 metadata 条件上传”逻辑。
- 已拒绝方案未在主线保留为默认行为。
