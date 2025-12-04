# LZO GPU 压缩/解压缩优化路线图

## 硬件信息
- **GPU**: Intel Iris Xe Graphics
- **计算单元**: 96 CUs
- **最大工作组**: 512
- **全局内存**: 28.71 GiB
- **本地内存**: 64 KiB per CU
- **时钟频率**: 1500 MHz
- **峰值带宽**: 37 GB/s (实测内部拷贝)

## 当前性能瓶颈分析

### 带宽利用率
```
峰值带宽: 37 GB/s
解压缩:   2.6 GB/s (7% 利用率)
压缩:     未详细测试
```
**结论**: 瓶颈不在带宽，在于计算/分支复杂度

### 并行度分析
```
当前配置:
  - 压缩: 每块 1 个 work-item (串行处理)
  - 解压缩: global_size=nblocks, local_size=8
  
理论上限:
  - 96 CUs × 512 threads = 49,152 并发线程
  - 当前最多使用: ~1000 threads (远低于上限)
```

---

## 优化策略 1: 提升并行度

### 1.1 解压缩并行化（当前最优先）

#### 问题
- 每个 block 只有 1 个 work-item 处理
- 块内串行，无法利用 GPU 并行能力

#### 方案：块内并行解压缩
```c
// 当前: 1 thread 处理整个 block
__kernel void lzo1x_decompress(__global const uchar *in, ...) {
    int bid = get_global_id(0);
    // 串行解压 block[bid]
}

// 优化后: N threads 协作处理 1 个 block
__kernel void lzo1x_decompress_parallel(
    __global const uchar *in,
    __local uchar *shared_buffer,  // 64KB 本地内存
    ...) {
    
    int bid = get_group_id(0);      // block ID
    int tid = get_local_id(0);       // thread ID in block
    int wg_size = get_local_size(0); // e.g., 64
    
    // 1. 协作加载压缩数据到 local memory (隐藏延迟)
    for (int i = tid; i < compressed_size; i += wg_size) {
        shared_buffer[i] = in[block_offset + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 2. 并行解压多个 literal/match 段
    //    - 预扫描找到所有 literal/match 边界
    //    - 每个 thread 处理一段
    //    - 需要 prefix-sum 计算输出偏移
}
```

**挑战**:
- LZO 压缩流高度依赖（后续指令依赖前面的输出位置）
- 需要预扫描或流水线设计

**收益**: 理论 10-50x（取决于块大小和依赖性）

---

### 1.2 压缩并行化

#### 当前瓶颈
```c
// lzo1x_1.cl: 完全串行
__kernel void lzo1x_compress(...) {
    int bid = get_global_id(0);
    // 逐字节扫描，查找字典匹配 - 完全串行
    for (ip = in_start; ip < in_end; ip++) {
        // 字典查找
        // 匹配长度计算
        // 输出 literal/match
    }
}
```

#### 方案：滑动窗口并行哈希
```c
__kernel void lzo1x_compress_parallel(
    __global const uchar *in,
    __local uint *dict_local,  // 本地字典（64KB）
    ...) {
    
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    
    // 1. 并行构建字典 (隐藏延迟)
    for (int i = tid; i < DICT_SIZE; i += wg_size) {
        dict_local[i] = EMPTY;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 2. 并行哈希多个位置
    //    每个 thread 负责 block 的一个区间
    int start = tid * (block_size / wg_size);
    int end = (tid + 1) * (block_size / wg_size);
    
    for (int i = start; i < end; i++) {
        uint hash = HASH(in[i], in[i+1], in[i+2]);
        // 并发更新字典（需要原子操作）
        atomic_max(&dict_local[hash], i);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 3. 主线程串行编码（或分段并行）
    if (tid == 0) {
        // 使用已构建的字典进行压缩
    }
}
```

**收益**: 字典构建加速 10-50x，整体加速 2-5x

---

## 优化策略 2: 延迟隐藏

### 2.1 异步数据传输（Overlap H2D/D2H with Compute）

#### 当前问题
```c
// main.c: 串行执行
clEnqueueWriteBuffer(queue, in_buf, CL_TRUE, ...);   // 阻塞上传
clEnqueueNDRangeKernel(queue, kernel, ...);          // 计算
clEnqueueReadBuffer(queue, out_buf, CL_TRUE, ...);   // 阻塞下载
```

#### 优化：流水线执行
```c
// 使用双缓冲 + 异步队列
cl_command_queue q_upload = clCreateCommandQueue(..., 0, ...);
cl_command_queue q_compute = clCreateCommandQueue(..., 0, ...);
cl_command_queue q_download = clCreateCommandQueue(..., 0, ...);

cl_mem buf_ping[2], buf_pong[2];  // 双缓冲

for (int i = 0; i < nblocks; i++) {
    int curr = i % 2;
    int next = (i + 1) % 2;
    
    // Stage 1: Upload block[i+1] (异步)
    if (i + 1 < nblocks) {
        clEnqueueWriteBuffer(q_upload, buf_ping[next], CL_FALSE, 
                             0, size[i+1], host_in[i+1], 0, NULL, &ev_upload);
    }
    
    // Stage 2: Compute block[i]
    clEnqueueNDRangeKernel(q_compute, kernel, ..., 
                           1, &ev_upload, &ev_compute);
    
    // Stage 3: Download block[i-1] (异步)
    if (i > 0) {
        clEnqueueReadBuffer(q_download, buf_pong[curr], CL_FALSE,
                            0, size[i-1], host_out[i-1], 1, &ev_compute, &ev_download);
    }
}
```

**收益**: 理论 30-50% 吞吐量提升（隐藏传输延迟）

---

### 2.2 指令级并行（ILP）优化

#### 解压缩 COPY_MATCH 优化
```c
// 当前: 串行 4 次 vload16
while (len >= 64) {
    uchar16 v0 = vload16(0, m_pos);
    uchar16 v1 = vload16(0, m_pos + 16);
    uchar16 v2 = vload16(0, m_pos + 32);
    uchar16 v3 = vload16(0, m_pos + 48);
    vstore16(v0, 0, op);
    vstore16(v1, 0, op + 16);
    vstore16(v2, 0, op + 32);
    vstore16(v3, 0, op + 48);
    op += 64; m_pos += 64; len -= 64;
}

// 优化: 交错 load/store，减少依赖
while (len >= 64) {
    uchar16 v0 = vload16(0, m_pos);
    uchar16 v1 = vload16(0, m_pos + 16);
    vstore16(v0, 0, op);          // 立即使用 v0
    uchar16 v2 = vload16(0, m_pos + 32);
    vstore16(v1, 0, op + 16);     // 立即使用 v1
    uchar16 v3 = vload16(0, m_pos + 48);
    vstore16(v2, 0, op + 32);
    vstore16(v3, 0, op + 48);
    op += 64; m_pos += 64; len -= 64;
}
```

**收益**: 5-10% (减少寄存器压力，改善流水线)

---

### 2.3 预取优化（Software Prefetch）

```c
// 解压缩主循环中预取下一个指令
__kernel void lzo1x_decompress(...) {
    __global const uchar *ip = in + offset;
    __global const uchar *ip_end = in + offset + size;
    
    // 预取前 4 个缓存行
    prefetch(ip, 4);
    
    while (ip < ip_end) {
        uchar t = *ip++;
        
        // 预取未来 64 字节
        if (ip + 64 < ip_end) {
            prefetch(ip + 64, 1);
        }
        
        // 处理指令...
    }
}
```

**注意**: OpenCL 2.0+ 支持 `prefetch()`，Intel GPU 支持

**收益**: 5-15%

---

## 优化策略 3: 内存层级优化

### 3.1 使用 Local Memory 缓存字典

#### 压缩优化
```c
__kernel void lzo1x_compress(
    __global const uchar *in,
    __global uchar *out,
    __local uint *dict_cache,  // 64KB 本地内存
    ...) {
    
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);
    
    // 初始化字典到 local memory
    for (int i = tid; i < DICT_SIZE; i += wg_size) {
        dict_cache[i] = EMPTY;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 使用 local memory 字典（访问延迟低）
    // ...
}
```

**收益**: 字典访问延迟从 ~200 cycles 降到 ~20 cycles

---

### 3.2 合并全局内存访问

```c
// 差的模式: 非合并访问
for (int i = 0; i < 16; i++) {
    out[i * stride] = data[i];  // 每个 thread 访问不连续地址
}

// 好的模式: 合并访问
int tid = get_global_id(0);
out[tid] = data[tid];  // 连续访问，硬件合并为一次事务
```

---

## 优化策略 4: 减少分支发散

### 4.1 解压缩指令解码优化

```c
// 当前: 多层嵌套 if-else (分支发散严重)
if (t >= 16) goto match;
if (t == 0) { /* 特殊处理 */ }
if (t >= 64) { /* M3 */ }
else if (t >= 32) { /* M4 */ }
else if (t >= 16) { /* M2 */ }

// 优化: 查表法减少分支
__constant uchar opcode_table[256] = {
    /* 预计算每个字节对应的操作类型 */
};

uchar opcode = opcode_table[t];
switch (opcode) {
    case OP_LITERAL: /* ... */ break;
    case OP_M2: /* ... */ break;
    case OP_M3: /* ... */ break;
    case OP_M4: /* ... */ break;
}
```

**收益**: 10-20% (减少 warp 内分支发散)

---

## 实施优先级

### 高优先级（短期，1-2周）
1. ✅ **解压缩循环展开**（已实现，+17.6%）
2. ⏳ **异步数据传输流水线**（收益 30-50%）
3. ⏳ **ILP 优化 COPY_MATCH**（收益 5-10%）
4. ⏳ **预取优化**（收益 5-15%）

### 中优先级（中期，2-4周）
5. ⏳ **压缩并行哈希字典**（收益 2-5x）
6. ⏳ **Local Memory 字典缓存**（收益 1.5-2x）
7. ⏳ **减少分支发散**（收益 10-20%）

### 低优先级（长期，1-2月）
8. ⏳ **解压缩块内并行**（收益 10-50x，实现复杂）
9. ⏳ **自适应块大小**（根据数据特征动态调整）

---

## 预期性能提升

### 解压缩
- 当前: 2.6 GB/s (7% 带宽利用率)
- 短期优化后: 5-8 GB/s (20% 利用率)
- 长期优化后: 15-25 GB/s (60% 利用率)

### 压缩
- 当前: 未测试，估计 ~1 GB/s
- 短期优化后: 2-3 GB/s
- 长期优化后: 5-10 GB/s

---

## 测试方法

### 微基准测试
```bash
# 延迟隐藏效果
./benchmark_async_transfer

# 并行度可扩展性
./benchmark_parallel_scale

# 内存层级性能
./benchmark_memory_hierarchy
```

### 端到端性能
```bash
# 49 个真实样本
for img in /root/samples/*.img; do
    ./lzo_gpu -c $img -o /tmp/test.lzo
    LZO_DECOMP_VEC=1 ./lzo_gpu -d /tmp/test.lzo -o /tmp/out.bin
done
```
