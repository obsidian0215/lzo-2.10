# Tools 目录脚本说明

本目录包含LZO项目的数据分析和可视化工具。

## 📁 目录结构

```
tools/
├── README.md                           # 本文档
│
├── 实验运行脚本 (Runners)
│   ├── run_lzo_cpu.sh                 # CPU基准测试
│   ├── run_lzo_gpu.sh                 # GPU实验管理器
│   ├── benchmark_hybrid.sh            # Hybrid模式基准
│   ├── param_scan.sh                  # GPU参数扫描
│   └── gpu_control.sh                 # GPU频率控制
│
├── 数据分析脚本 (Analysis)
│   ├── analyze.py                     # GPU结果分析 (主要)
│   ├── aggregate_results.py           # CPU跨配置聚合
│   └── parse_profile_logs.py          # GPU profiling分析
│
├── 可视化脚本 (Plotting)
│   ├── plot_gpu_analysis.py           # GPU综合图表 (统一)
│   ├── plot_aggregate.py              # CPU配置热力图
│   └── plot_throughput_freq_threads.py # CPU频率曲线
│
└── 工具脚本 (Utilities)
    ├── aggregate_experiment_markdowns.py  # 汇总实验报告
    └── train_performance_model.py         # 性能模型训练
```

## 🚀 快速开始

### GPU性能分析完整流程

```bash
# 1. 运行参数扫描
./param_scan.sh

# 2. 分析结果
./analyze.py

# 3. 生成图表和报告
./plot_gpu_analysis.py

# 4. 查看报告
cat ../exp_results/lzo_gpu/logs/REPORT.md
```

### CPU性能分析

```bash
# 1. 运行基准测试
./run_lzo_cpu.sh -s ../samples/test_1mb.dat -a 1k -t 4

# 2. 聚合多个配置
./aggregate_results.py

# 3. 生成热力图
./plot_aggregate.py
```

## 📊 核心脚本详解

### 1. 实验运行脚本

#### `run_lzo_cpu.sh`
CPU压缩基准测试脚本。

**功能：**
- 单次或批量测试
- 多算法支持 (1, 1k, 1l, 1o)
- 多线程支持 (1-8线程)
- CPU频率控制
- 自动生成summary.csv

**用法：**
```bash
# 单次测试
./run_lzo_cpu.sh -s sample.dat -a 1k -t 4

# 批量测试 (所有算法, 多线程)
./run_lzo_cpu.sh -s sample.dat -A "1,1k,1l" -T "1,2,4,8"
```

**输出：**
- `exp_results/lzo_cpu/<config>/summary.csv`
- 包含压缩比、吞吐量、时间等指标

---

#### `param_scan.sh`
GPU参数空间扫描脚本。

**功能：**
- 扫描workgroup_size (32, 64, 128, 256)
- 扫描vector_length (1, 2, 4, 8)
- 测试多个压缩variant
- 生成详细日志

**用法：**
```bash
# 完整参数扫描
./param_scan.sh

# 自定义参数
./param_scan.sh --wg "64,128" --vlen "1,4"
```

**输出：**
- `exp_results/lzo_gpu/logs/param_scans/*.log`
- 每个配置的压缩/解压性能

---

#### `benchmark_hybrid.sh`
Hybrid模式(CPU+GPU)基准测试。

**功能：**
- 测试不同CPU线程 + GPU组合
- 自动选择最优调度策略
- 生成性能对比报告

**用法：**
```bash
./benchmark_hybrid.sh -s sample.dat -c 2 -g 1
```

---

### 2. 数据分析脚本

#### `analyze.py` ⭐ (GPU主力分析工具)
GPU实验结果综合分析工具。

**功能：**
- 聚合所有param_scan日志
- 计算统计量 (mean, median, stdev)
- 生成summary.csv和analysis_summary.csv
- 对比向量化 vs 标量性能
- 生成文本报告

**用法：**
```bash
# 使用默认路径
./analyze.py

# 自定义输入/输出
./analyze.py -i logs/param_scans -o results/summary.csv
```

**输出：**
- `exp_results/lzo_gpu/logs/summary.csv` (详细per-run数据)
- `exp_results/lzo_gpu/logs/analysis_summary.csv` (聚合统计)
- 终端打印分析报告

**输出示例：**
```csv
core,comp,wg,vlen,n_samples,avg_comp_MBps,avg_decomp_MBps,avg_ratio
lzo1x_1,1,128,4,5,156.3,487.2,2.13
lzo1x_1k,1k,64,1,5,201.5,512.8,2.01
```

---

#### `aggregate_results.py`
CPU跨配置结果聚合。

**功能：**
- 读取多个summary.csv
- 按配置分组统计
- 生成aggregate_results.csv

**用法：**
```bash
./aggregate_results.py
```

**输入：**
- `exp_results/lzo_cpu/*/summary.csv`

**输出：**
- `exp_results/lzo_cpu/aggregate_results.csv`

---

#### `parse_profile_logs.py`
GPU kernel profiling日志解析。

**功能：**
- 提取OpenCL profiling信息
- 分析kernel执行时间
- 计算数据传输开销

**用法：**
```bash
./parse_profile_logs.py -i profiles/ -o summary_profiles.csv
```

---

### 3. 可视化脚本

#### `plot_gpu_analysis.py` ⭐ (GPU统一绘图工具)
GPU性能分析综合可视化工具（合并了原generate_plots.py和generate_throughput_plots.py）。

**功能：**
- 压缩比直方图
- 压缩 vs 解压散点图
- Top-15配置柱状图
- 吞吐量分布直方图
- 生成Markdown报告 (REPORT.md)

**用法：**
```bash
# 使用默认路径
./plot_gpu_analysis.py

# 自定义输入
./plot_gpu_analysis.py -i path/to/analysis_summary.csv
```

**输出：**
- `exp_results/lzo_gpu/logs/plots/*.png` (6个图表)
- `exp_results/lzo_gpu/logs/REPORT.md` (带图表的报告)

**生成的图表：**
1. `comp_ratio_hist.png` - 压缩比分布
2. `comp_vs_decomp_scatter.png` - 压缩/解压性能散点图
3. `top_comp_throughput.png` - Top-15压缩配置
4. `top_decomp_throughput.png` - Top-15解压配置
5. `hist_comp_throughput.png` - 压缩吞吐量分布
6. `hist_decomp_throughput.png` - 解压吞吐量分布

---

#### `plot_aggregate.py`
CPU配置热力图生成。

**功能：**
- 生成频率×线程数热力图
- 对比不同governor设置
- 显示性能热点

**用法：**
```bash
./plot_aggregate.py
```

**输出：**
- `exp_results/lzo_cpu/plots/heatmap_*.png`

---

#### `plot_throughput_freq_threads.py`
CPU频率扫描曲线图。

**功能：**
- 绘制吞吐量 vs CPU频率曲线
- 区分不同线程数
- 显示turbo开关影响

**用法：**
```bash
./plot_throughput_freq_threads.py
```

**输出：**
- `exp_results/lzo_cpu/plots/throughput_freq_threads_combined.png`

---

### 4. 工具脚本

#### `aggregate_experiment_markdowns.py`
汇总所有实验的Markdown报告。

**用法：**
```bash
./aggregate_experiment_markdowns.py
```

**输出：**
- `EXPERIMENT_SUMMARY_ALL.md`

---

#### `train_performance_model.py`
基于实验数据训练性能预测模型。

**功能：**
- 读取历史实验数据
- 训练回归模型 (线性/多项式)
- 预测新配置性能

**用法：**
```bash
./train_performance_model.py -i summary.csv -o model.pkl
```

---

## 🔄 典型工作流

### GPU性能调优工作流

```bash
# 1. 运行完整参数扫描
cd tools
./param_scan.sh

# 2. 分析结果
./analyze.py
# 输出: exp_results/lzo_gpu/logs/analysis_summary.csv

# 3. 生成可视化
./plot_gpu_analysis.py
# 输出: exp_results/lzo_gpu/logs/plots/*.png
#       exp_results/lzo_gpu/logs/REPORT.md

# 4. 查看最佳配置
head -20 ../exp_results/lzo_gpu/logs/REPORT.md
```

### CPU性能测试工作流

```bash
# 1. 运行频率扫描
./run_lzo_cpu.sh -s sample.dat -F "800,1600,2400" -T "1,4,8"

# 2. 聚合结果
./aggregate_results.py

# 3. 生成热力图
./plot_aggregate.py

# 4. 生成频率曲线
./plot_throughput_freq_threads.py
```

### Hybrid模式评估

```bash
# 1. 运行Hybrid基准
./benchmark_hybrid.sh -s large_file.dat -c 4 -g 1

# 2. 对比CPU-only和GPU-only
./run_lzo_cpu.sh -s large_file.dat -t 4
./run_lzo_gpu.sh compress large_file.dat output.lzo

# 3. 分析性能提升
# (查看benchmark_hybrid.sh生成的报告)
```

---

## 📝 最佳实践

### 1. 数据组织

所有实验结果统一放在：
```
exp_results/
├── lzo_cpu/
│   ├── config1/summary.csv
│   ├── config2/summary.csv
│   └── aggregate_results.csv
│
├── lzo_gpu/
│   └── logs/
│       ├── param_scans/*.log
│       ├── summary.csv
│       ├── analysis_summary.csv
│       ├── REPORT.md
│       └── plots/*.png
│
└── lzo_hybrid/
    └── benchmark_results.csv
```

### 2. 脚本执行顺序

**GPU分析：**
```
param_scan.sh → analyze.py → plot_gpu_analysis.py
```

**CPU分析：**
```
run_lzo_cpu.sh → aggregate_results.py → plot_aggregate.py
```

### 3. 版本控制

- ✅ 提交脚本源码
- ✅ 提交REPORT.md
- ❌ 不提交CSV数据文件
- ❌ 不提交PNG图表

`.gitignore` 应包含：
```
exp_results/*.csv
exp_results/**/*.png
exp_results/**/*.log
```

---

## 🐛 故障排查

### 问题1: analyze.py找不到日志

**症状：**
```
ERROR: No log files found in exp_results/lzo_gpu/logs/param_scans
```

**解决：**
```bash
# 先运行参数扫描
./param_scan.sh
```

---

### 问题2: plot_gpu_analysis.py报matplotlib错误

**症状：**
```
ERROR: matplotlib not available
```

**解决：**
```bash
pip3 install matplotlib numpy
```

---

### 问题3: 权限错误

**症状：**
```
Permission denied: ./analyze.py
```

**解决：**
```bash
chmod +x tools/*.py tools/*.sh
```

---

## 📚 参考文档

- **LZO GPU实现**: `lzo_gpu/PERFORMANCE_SUMMARY.md`  (合并后的实现与性能说明)
- **Hybrid模式设计**: `/root/lzo-2.10/docs/HYBRID_DESIGN.md`
- **LZ4 GPU分析**: `/root/lz4/LZ4_GPU_ANALYSIS.md`

---

## 🔧 维护说明

### 已整理的脚本

- ✅ 删除 `summarize_throughput.py` (功能被analyze.py覆盖)
- ✅ 合并 `generate_plots.py` + `generate_throughput_plots.py` → `plot_gpu_analysis.py`
- ✅ 保留专业化脚本 (CPU热力图、频率曲线等)

### 脚本清单

| 脚本 | 状态 | 说明 |
|------|------|------|
| analyze.py | ✅ 活跃 | GPU主力分析工具 |
| plot_gpu_analysis.py | ✅ 活跃 | GPU统一绘图工具 (新) |
| aggregate_results.py | ✅ 活跃 | CPU聚合工具 |
| run_lzo_cpu.sh | ✅ 活跃 | CPU基准测试 |
| param_scan.sh | ✅ 活跃 | GPU参数扫描 |
| summarize_throughput.py | ❌ 已删除 | 被analyze.py替代 |
| generate_plots.py | ❌ 已删除 | 合并到plot_gpu_analysis.py |
| generate_throughput_plots.py | ❌ 已删除 | 合并到plot_gpu_analysis.py |

---

**文档版本**: 2.0
**最后更新**: 2024-11-22
**维护者**: LZO Tools Team
