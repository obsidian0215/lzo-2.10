# lzo_hybrid（完整实现说明）

`lzo_hybrid` 是 `lzo-2.10` 工作区中的 CPU+GPU 协同主程序，负责将 `lzo_cpu` 与 `lzo_gpu` 两条执行路径组织为统一的可验证压缩/解压引擎。其职责不仅是调用内核，还包括任务分配、并行调度、容器组织、基准统计与正确性闭环。

本文档按“完整工程说明书”组织，提供可复现实验所需的核心信息。

## 1. 模块目标与能力边界

核心目标：

1. 在块级任务模型下并行利用 CPU 与 GPU；
2. 支持固定比例与自适应比例两类分工策略；
3. 支持 `lzo1x` 与 `lzo1y` 算法线；
4. 保持压缩容器在解压端可确定性回放；
5. 提供可复现的 `--bench` 基准输出和校验路径。

能力边界：

- 当前主实现以 Linux 路径为主，Windows 重点支持 standalone 与 benchmark；
- 性能结论必须与验证结果绑定，任何未通过 roundtrip 的结果不得用于结论。

## 2. 构建说明

### 2.1 Linux

```bash
make -C ../lzo_gpu
make
```

依赖：

- C 编译器（推荐 `gcc`）
- OpenCL 头文件与运行时加载器
- `pthread`

### 2.2 Windows（MSYS2 / MinGW-w64）

```bash
make -C ../lzo_gpu \
  OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"

make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

补充：

- 如 OpenCL 导入库名不同，可设置 `OPENCL_LIB_NAME`；
- OpenCL 运行时需由系统驱动提供可加载 DLL/SO。

## 3. 使用方式

### 3.1 基本命令

```bash
./lzo_hybrid input.bin -o out.lzo
./lzo_hybrid -d out.lzo -o restored.bin
```

### 3.2 核心参数（完整）

- `-a`, `--alg`：算法（`lzo1x` 或 `lzo1y`）
- `-L`, `--level`：等级/字典位（11~16，99，999）
- `-B`, `--block-size`：固定块大小
- `--cpu-threads`：CPU 线程数（默认自动探测）
- `--gpu-ratio`：固定 GPU 比例（0.0=纯 CPU，1.0=纯 GPU）
- `--adaptive`：启用自适应分配
- `--sample-blocks`：自适应采样块数
- `--local`：OpenCL local work-group size
- `--bench N`：进程内预热后执行 N 轮

默认不显式指定 `-B` 时，会进入自适应块选择流程，并叠加最小块保护。

### 3.3 基准示例

```bash
./lzo_hybrid --bench 3 -B 64K --cpu-threads 2 --gpu-ratio 0.7 /path/to/file
./lzo_hybrid --bench 3 -B 64K --cpu-threads 2 --adaptive --sample-blocks 8 /path/to/file
```

## 4. 容器与回放机制

`lzo_hybrid` 容器保存以下信息：

1. 基础头部（magic/版本/算法标识）；
2. 块总数、块大小、GPU 分配块统计；
3. 每块压缩长度；
4. GPU/CPU 两路数据段。

解压阶段按容器元信息严格重建任务分配并逐块恢复，保证压缩/解压闭环一致。

## 5. 执行架构与实现细节

### 5.1 CPU 子路径

- 采用 `liblzo2 + pthread`；
- 工作线程并行处理块任务；
- 对于高压缩等级路径，保持算法语义一致；
- 线程自动探测可覆盖大多数默认部署场景。

### 5.2 GPU 子路径

- 采用 OpenCL 内核执行压缩与解压；
- workspace 生命周期按“可复用、按需增长”管理；
- 通过参数缓存与缓冲复用减少 steady-state 调度成本；
- 在纯 GPU 比例下尽量削减主机侧不必要开销。

### 5.3 协同并行路径

- 输入数据在块级完成 CPU/GPU 分工；
- 两路并发执行，最终在统一容器合并；
- 解压端依据相同元信息进行逆向回放。

## 6. 自适应模型（完整）

`--adaptive` 路径通过多维信号估计最优 GPU 比例：

1. 校准 CPU/GPU 基准吞吐；
2. 结合样本估计压缩率对吞吐的影响；
3. 使用运行时负载评估有效可用算力；
4. 在总工期最小化约束下输出比例；
5. 对小输入场景应用保护回退，避免固定开销吞噬收益。

模型默认按确定性配置运行，以控制验证空间并提高回归稳定性。

## 7. 计时语义与数据解释

### 7.1 Kernel 指标

用于观察编解码核心执行跨度与算力上限，不直接代表端到端吞吐。

### 7.2 Total 指标

用于观察工程真实表现，覆盖运行时协调成本，适合作为回归门禁与版本对比依据。

### 7.3 正确性门禁

性能统计必须以 roundtrip 验证通过为前置条件；失败样本应单独标注并排除结论。

## 8. 环境变量

- `FORCE_OPENCL_DEVICE=GPU|CPU|DEFAULT|ALL`
- `LZO_STANDARD_COPY=0|1`

建议在批量实验中固定环境变量并随结果工件一起记录。

## 9. 复现实验建议

1. 固定输入样本集与文件顺序；
2. 固定算法、块大小、线程数与比例策略；
3. 记录可执行文件哈希；
4. 每轮执行压缩与解压并做文件级校验；
5. 输出均值/中位数/分位数，避免只看单点。

## 10. 排障说明

1. **OpenCL 不可用**：优先检查驱动、设备可见性、库路径；
2. **吞吐离散度大**：确认是否混入冷启动、系统负载与温控降频；
3. **线程放大无收益**：检查 CPU 绑定与内存带宽瓶颈；
4. **结果不可复现**：检查二进制是否变化、参数是否漂移、样本是否一致。

## 11. 当前版本使用结论

- `lzo_hybrid` 已具备完整的 CPU/GPU 协同执行与验证能力；
- 固定与自适应两种分配策略均可用于正式测试；
- 工程实践中应以“正确性稳定、口径一致、可复现”作为首要约束；
- 文档长期与代码行为对齐，所有性能结论应绑定可追溯工件。
