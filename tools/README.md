# tools

本目录放 LZO 测试、功耗/频率 wrapper、硬件遥测和控制脚本。当前主测试入口是 `bench_lzo.py`；频率扫描和功耗采样不再写进 `bench_lzo.py`，由外部 wrapper 调用。

## 脚本职责

| 脚本 | 职责 | 输出 |
| --- | --- | --- |
| `bench_lzo.py` | 只做压缩/解压性能测试；默认 `1` 轮 bench + `6` 轮真实压缩/解压 | `raw.csv`、`per_file_summary.csv`、`aggregate.csv` |
| `bench_lzo_power_wrapper.py` | 外部频率点循环、功耗/频率采样、调用 `bench_lzo.py` | `power_frequency_summary.csv` + 每个频率点的 bench 子目录 |
| `hw_telemetry.py` | CPU/GPU/DRAM 功耗与频率采样接口 | 被 wrapper 调用 |
| `cpu_control.sh` | Linux CPU 频率/核心/模式控制和 reset | 被 wrapper 调用 |
| `gpu_control.sh` | Linux GPU 频率/功耗/模式控制和 reset | 被 wrapper 调用 |
| `generate-test-data.py` | 生成合成测试数据 | 手动使用 |

`lzop` 不再作为默认对比对象；`lzo_cpu` 的 native CPU 实现已经可以作为 CPU baseline。`bench_lzo.py` 仍保留 `lzop` 引擎入口，只有显式传 `--engines lzop` 时才会运行。`opencl_cpu` 也不进默认矩阵，因为它本质上等价于 `hybrid` 的 `gpu_ratio=0`。

## 构建前置

Windows 使用 MSYS2/UCRT64 时，需要显式设置 PATH：

```powershell
$env:PATH='C:\msys64\ucrt64\bin;C:\msys64\usr\bin;' + $env:PATH
mingw32-make -C lzo_cpu win32-mingw
mingw32-make -C lzo_gpu lzo_gpu
mingw32-make -C lzo_hybrid lzo_hybrid
```

Linux：

```bash
make -C lzo_cpu gcc
make -C lzo_gpu lzo_gpu
make -C lzo_hybrid lzo_hybrid
```

## `bench_lzo.py`

### 默认扫描列表

`bench_lzo.py` 顶部有一组默认扫描矩阵，方便直接改标准测试范围：

```python
DEFAULT_ENGINES = ["gpu", "native_cpu", "hybrid"]
DEFAULT_ALG = "lzo1x"
DEFAULT_GPU_LEVELS = [15]
DEFAULT_CPU_LEVELS = [14]
DEFAULT_HYBRID_GPU_LEVELS = [15]
DEFAULT_HYBRID_CPU_LEVELS = [14]
DEFAULT_GPU_BLOCKS = ["48KB", "64KB"]
DEFAULT_CPU_BLOCKS = ["64KB"]
DEFAULT_HYBRID_BLOCKS = ["48KB", "64KB"]
DEFAULT_LOCAL_SIZES = [1]
DEFAULT_CPU_THREADS = [1]
DEFAULT_GPU_RATIOS = [0, 0.5]
DEFAULT_LZOP_LEVELS = [1, 3, 5]
DEFAULT_BENCH_SECONDS = 5
DEFAULT_MANUAL_ROUNDS = 6
```

命令行参数会覆盖这些默认值。`lzop` 保留在可选引擎里，但不在 `DEFAULT_ENGINES` 中。

### 默认口径

- `--bench-seconds 5`
- `--manual-rounds 6`
- 默认引擎：`gpu,native_cpu,hybrid`
- bench 阶段：调用程序自己的 `--bench`，作为缺少 manual kernel 字段时的补充来源。
- manual 阶段：真实压缩到临时文件、真实解压、校验 hash，收集主吞吐与端到端吞吐。
- 临时产物：写入本次 run 的 `tmp/`，结束时 `rmtree`，不污染样本目录。
- `hybrid` 默认测试 `gpu_ratio=0,0.5`；`0` 是 OpenCL CPU-only，`0.5` 是 GPU/CPU 固定比例混合。
- `--gpu-ratios adaptive` 会传给 `lzo_hybrid --adaptive`；当前实现用 50/50 作为自适应入口的初始策略，后续自适应模型在此入口上扩展。
- `hybrid` 支持 `--hybrid-gpu-levels` 和 `--hybrid-cpu-levels`，用于给 GPU/CPU 压缩侧加载不同 `D_BITS` kernel；解压侧按容器块元数据解码，不需要保存 per-block level。

### 常用调用

全样本默认 GPU + native CPU + hybrid：

```bash
python tools/bench_lzo.py \
  --platform-id windows_arc \
  --samples C:\Users\obsid\git-repo\samples
```

Linux 全样本：

```bash
python3 tools/bench_lzo.py \
  --platform-id linux_xe \
  --samples /root/samples
```

只测 native CPU，线程数 `1/2`，block `48/64KB`：

```bash
python tools/bench_lzo.py \
  --engines native_cpu \
  --cpu-threads 1,2 \
  --cpu-blocks 48KB,64KB \
  --samples C:\Users\obsid\git-repo\samples
```

只测 GPU，block `48/64KB`，D_BITS `13/14/15`：

```bash
python tools/bench_lzo.py \
  --engines gpu \
  --gpu-blocks 48KB,64KB \
  --gpu-levels 13,14,15 \
  --samples C:\Users\obsid\git-repo\samples
```

显式测试 OpenCL CPU：

```bash
python tools/bench_lzo.py \
  --engines opencl_cpu \
  --cpu-threads 1,2 \
  --cpu-blocks 48KB,64KB \
  --gpu-levels 14 \
  --samples C:\Users\obsid\git-repo\samples
```

等价的推荐写法是用 hybrid CPU-only：

```bash
python tools/bench_lzo.py \
  --engines hybrid \
  --gpu-ratios 0 \
  --cpu-threads 1,2 \
  --hybrid-blocks 48KB,64KB \
  --samples C:\Users\obsid\git-repo\samples
```

显式测试 hybrid 固定比例：

```bash
python tools/bench_lzo.py \
  --engines hybrid \
  --gpu-ratios 0,0.25,0.5,0.75,1 \
  --cpu-threads 1,2 \
  --hybrid-blocks 48KB,64KB \
  --samples C:\Users\obsid\git-repo\samples
```

测试 hybrid adaptive 入口：

```bash
python tools/bench_lzo.py \
  --engines hybrid \
  --gpu-ratios adaptive \
  --cpu-threads 1 \
  --hybrid-blocks 48KB,64KB \
  --samples C:\Users\obsid\git-repo\samples
```

### 输出

每次运行生成：

```text
exp_results/runs/<timestamp>/
  raw.csv
  per_file_summary.csv
  aggregate.csv
  run_meta.txt
```

字段含义：

- `comp_mbs_*` / `dec_mbs_*`：压缩/解压主吞吐；优先来自真实 manual 阶段的 kernel/span 字段，缺失时回退到 bench 阶段。
- `e2e_comp_mbs_*` / `e2e_dec_mbs_*`：端到端吞吐；真实压缩/解压路径，OpenCL 路径排除 OpenCL init/build。
- `ratio_pct`：压缩率。
- `verify_ok` / `verify_all`：解压 hash 校验结果。
- `gpu_level` / `cpu_level`：hybrid 压缩侧分别使用的 `D_BITS`；普通 GPU/CPU 引擎为空。

## `bench_lzo_power_wrapper.py`

该脚本是外部 wrapper：设置频率点、运行 `bench_lzo.py`、采样功耗/频率、写独立 CSV。它不修改 `bench_lzo.py` 的三表格式。

### 关键规则

- 每个频率点执行前可 reset；默认会 reset。
- 每个频率点结束后必定在 `finally` 中 reset CPU/GPU，避免 CPU/GPU 一直被限制。
- Windows 下频率控制脚本返回 `unsupported_on_windows`，但 wrapper 仍可用 Windows Energy Meter 做功耗采样。
- Linux 下频率控制通常需要 root 或免密 sudo。

### 固定 MHz 频率点示例

```bash
python3 tools/bench_lzo_power_wrapper.py \
  --platform-id linux_xe \
  --cpu-frequencies 2100,3400,NA \
  --gpu-frequencies 1000,NA \
  --output-dir exp_results/power_freq_runs/linux_xe_scan \
  -- \
  --engines gpu,native_cpu \
  --samples /root/samples \
  --bench-seconds 5 \
  --manual-rounds 6
```

### 百分比频率点示例

```bash
python3 tools/bench_lzo_power_wrapper.py \
  --platform-id linux_xe \
  --cpu-frequencies 40%,70%,100%,NA \
  --gpu-frequencies 40%,70%,100%,NA \
  --output-dir exp_results/power_freq_runs/linux_xe_percent \
  -- \
  --engines gpu,native_cpu \
  --samples /root/samples
```

频点解析规则：

- `NA` / `NONE` / `-`：不限制该设备；wrapper 会先调用对应 `control.sh reset` 清空限制。
- `2100` 或 `2100MHz`：固定到绝对 MHz。
- `70%`：调用 `control.sh freq 70`；百分比由控制脚本按设备最大/最小频率换算。
- CPU 和 GPU 是两个独立列表，但 wrapper 会按 workload 过滤不相关频率：
  - `gpu`：只扫描 GPU 频率，CPU 固定为 `NA/reset`；
  - `native_cpu` / `opencl_cpu` / `lzop`：只扫描 CPU 频率，GPU 固定为 `NA/reset`；
  - `hybrid` 且 `gpu_ratio=0`：OpenCL CPU-only，只扫描 CPU 频率；
  - `hybrid` 且 `gpu_ratio=1`：GPU-only，只扫描 GPU 频率；
  - `hybrid` 且 `0<gpu_ratio<1` 或 `adaptive`：扫描 CPU×GPU 频率矩阵。
- 旧的成对 `pairs` 接口已经废弃；如果需要固定某个设备不限制，直接在对应列表中只保留 `NA`。

### 输出

```text
exp_results/power_freq_runs/<run>/
  power_frequency_summary.csv
  point_001_cpu_<...>_gpu_<...>/
    wrapper_command.txt
    wrapper_output.log
    runs/<timestamp>/
      raw.csv
      per_file_summary.csv
      aggregate.csv
```

`power_frequency_summary.csv` 包含：

- CPU/GPU 频率目标、目标类型与 apply/reset 状态；
- bench 子目录；
- CPU/GPU/DRAM energy；
- CPU/GPU 平均频率；
- CPU/GPU/DRAM 平均功耗与峰值功耗。

## 说明

- `bench_lzo.py` 不做功耗、频率、系统限制控制。
- `bench_lzo_power_wrapper.py` 不解释性能结果，只负责外部实验条件和遥测记录。
- 如果要做新的设备自适应或调度策略扫描，优先通过 wrapper 组合参数，不要再把复杂实验矩阵塞回 `bench_lzo.py`。
