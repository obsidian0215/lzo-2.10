# LZO GPU 向量化解压缩优化分析

## 问题回顾：COPY_MATCH 的挑战

LZO 解压缩中的 `COPY_MATCH(op, m_pos, len)` 是性能关键路径，需要将 `len` 字节从 `m_pos` 复制到 `op`。核心挑战是**重叠复制**：

```
情况1: offset >= len (无重叠)
  m_pos: [A][B][C][D][ ][ ][ ][ ]
  op:    [ ][ ][ ][ ][?][?][?][?]
  
情况2: offset < len (有重叠 - RLE 模式)
  offset=1, len=8
  m_pos: [X]
  op:    [X][X][X][X][X][X][X][X]
```

## 已发现的 Bug

### Bug 1: offset==4 尾部处理错误 (已修复)

**错误代码**:
```c
while (len--) *op++ = m_pos[len % 4];  // BUG!
```

**问题**: `len--` 在每次迭代中递减，导致 `len % 4` 访问错误的索引
- 迭代1: len=3, 访问 m_pos[3] (应该 m_pos[0])
- 迭代2: len=2, 访问 m_pos[2] (应该 m_pos[1])
- 迭代3: len=1, 访问 m_pos[1] (应该 m_pos[2])

**修复方案**:
```c
if (len >= 1) *op++ = p0;
if (len >= 2) *op++ = p1;
if (len >= 3) *op++ = p2;
```

## 当前优化策略

### 1. 无重叠优化 (offset >= len)
```c
if (offset >= len) {
    while (len >= 16) { vstore16(vload16(0, m_pos), 0, op); op+=16; m_pos+=16; len-=16; }
    if (len >= 8)     { vstore8(vload8(0, m_pos), 0, op); op+=8; m_pos+=8; len-=8; }
    while (len--)     { *op++ = *m_pos++; }
}
```
**效果**: 1 次 vload16 = 16 字节加载（vs 16 次标量加载），最高 **16x 加速**

### 2. RLE 模式优化 (offset=1/2/4/8)

#### offset=1 (单字节重复)
```c
uchar c = *m_pos;
uchar16 fill = (uchar16)(c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c);
while (len >= 16) { vstore16(fill, 0, op); op+=16; len-=16; }
```
**效果**: 1 次向量 store = 16 字节（vs 16 次标量 store）

#### offset=2/4/8 (模式重复)
同理，预构造重复模式向量，批量写入

### 3. 一般重叠 (offset > 8, offset < len)
```c
while (len--) { *op++ = *m_pos++; }  // 回退到标量逐字节
```

## 性能分析

### 测试结果 (49 个真实样本)
- **标量内核**: 90-1241 MB/s (平均 ~500 MB/s)
- **向量内核**: 157-2184 MB/s (平均 ~900 MB/s)
- **加速比**: 1.4x - 2.2x

### 关键样本性能
```
redis-video_parent_6 (107 MB):
  标量: 1241 MB/s
  向量: 2184 MB/s  → 1.76x 加速

redis-video_parent_5 (76 MB):
  标量:  972 MB/s
  向量: 1452 MB/s  → 1.49x 加速
```

## 进一步优化空间

### 优化方向 1: 扩展 RLE 检测范围

**当前限制**: 只检测 offset=1/2/4/8
**问题**: offset=3/5/6/7/9-15 等模式未优化

**提案**: 扩展到 offset <= 16 的所有模式
```c
if (offset <= 16 && offset < len) {
    // 动态构造重复模式
    uchar pattern[16];
    for (int i = 0; i < 16; i++) pattern[i] = m_pos[i % offset];
    
    while (len >= 16) {
        vstore16(vload16(0, pattern), 0, op);
        op += 16; len -= 16;
    }
    // ... 剩余处理
}
```

**收益**: 
- ✅ 覆盖更多压缩模式
- ❌ 动态构造有额外开销
- **预期**: 对 offset=3/5/6/7 等常见模式有 5-10% 提升

---

### 优化方向 2: 中等重叠的"滑动窗口"向量化

**当前问题**: offset=9-15 时，回退到标量
**思路**: 使用"部分重叠"向量拷贝

```c
if (offset >= 8 && offset < 16 && len > offset) {
    // 第一轮：复制 offset 字节
    for (int i = 0; i < offset; i++) *op++ = *m_pos++;
    len -= offset;
    
    // 现在 offset 已扩大，可安全向量化
    while (len >= 16) {
        vstore16(vload16(0, m_pos), 0, op);
        op += 16; m_pos += 16; len -= 16;
    }
    // ... 尾部处理
}
```

**示例**: offset=10, len=100
1. 先标量复制 10 字节
2. 现在 op-m_pos=10，剩余 90 字节
3. 用向量化处理剩余 80 字节（5次 vload16）
4. 标量处理最后 10 字节

**收益**: 
- ✅ offset=9-15 时避免完全标量化
- **预期**: 这类场景 20-40% 提升

---

### 优化方向 3: 非对齐向量加载优化

**当前问题**: `vload16` 要求地址对齐，非对齐时性能下降
**OpenCL 特性**: `vload` 系列函数处理非对齐，但可能有隐藏开销

**检测对齐并优化**:
```c
if (offset >= len) {
    // 检查对齐
    if (((uintptr_t)m_pos & 15) == 0 && ((uintptr_t)op & 15) == 0) {
        // 对齐路径：直接用原生指针
        while (len >= 16) {
            *((__global uchar16*)op) = *((__global const uchar16*)m_pos);
            op += 16; m_pos += 16; len -= 16;
        }
    } else {
        // 非对齐路径：使用 vload/vstore
        while (len >= 16) {
            vstore16(vload16(0, m_pos), 0, op);
            // ...
        }
    }
}
```

**收益**: 
- ✅ 对齐情况下减少 vload/vstore 开销
- ❌ 分支增加代码复杂度
- **预期**: 对齐场景 5-15% 提升

---

### 优化方向 4: 预取优化 (GPU 特定)

**思路**: 提前触发内存加载，掩盖延迟
```c
if (offset >= len && len >= 64) {
    // 预取前 2 个缓存行
    prefetch(m_pos, 2);
    
    while (len >= 32) {
        prefetch(m_pos + 64, 2);  // 预取下一轮
        vstore16(vload16(0, m_pos), 0, op);
        vstore16(vload16(0, m_pos+16), 0, op+16);
        op += 32; m_pos += 32; len -= 32;
    }
    // ...
}
```

**注意**: OpenCL 没有标准 prefetch，需设备特定扩展
**收益**: GPU 上可能 10-20% 提升（取决于硬件）

---

### 优化方向 5: 展开循环减少分支

**当前**: 每 16 字节一次循环判断
**优化**: 展开 4 次迭代（64 字节/批次）

```c
if (offset >= len) {
    while (len >= 64) {
        vstore16(vload16(0, m_pos), 0, op);
        vstore16(vload16(0, m_pos+16), 0, op+16);
        vstore16(vload16(0, m_pos+32), 0, op+32);
        vstore16(vload16(0, m_pos+48), 0, op+48);
        op += 64; m_pos += 64; len -= 64;
    }
    while (len >= 16) { /* ... */ }
    // ...
}
```

**收益**: 
- ✅ 减少循环开销和分支预测失败
- **预期**: 5-10% 提升（大块传输场景）

---

### 优化方向 6: 自适应策略（根据 len 选择路径）

**思路**: 小块用标量，大块用向量
```c
if (offset >= len) {
    if (len < 32) {
        // 小块：标量更快（避免向量设置开销）
        while (len--) *op++ = *m_pos++;
    } else {
        // 大块：向量化
        while (len >= 16) { /* vload16 */ }
        // ...
    }
}
```

**收益**: 
- ✅ 避免小块时向量化 overhead
- **预期**: 小块场景 10-20% 提升

---

## 风险与权衡

### 已知陷阱
1. **重叠语义**: 必须保证逐字节 memmove 语义
2. **尾部边界**: len 不是 16 倍数时的剩余处理
3. **模式构造开销**: 动态生成向量可能比标量慢

### 安全优化原则
1. **保守检测**: 只有 100% 确定无重叠时才向量化
2. **小块回退**: len < 阈值时用标量
3. **充分测试**: 用真实样本验证每个优化

### 建议优化顺序
1. **优先**: 优化方向 2（中等重叠滑动窗口）- 收益大，风险低
2. **次要**: 优化方向 5（循环展开）- 简单有效
3. **实验**: 优化方向 1（扩展 RLE）- 需测试开销
4. **高级**: 优化方向 3（对齐优化）- 复杂度高

## 总结

当前实现已实现：
- ✅ 无重叠场景 16 字节向量化
- ✅ offset=1/2/4/8 RLE 模式优化
- ✅ 49/49 样本正确性验证
- ✅ 平均 1.5-2x 性能提升

剩余优化空间：
- 🔧 offset=3/5/6/7/9-15 等模式（预期 10-15% 整体提升）
- 🔧 循环展开和预取（预期 5-10% 提升）
- 🔧 小块自适应策略（预期 5% 提升）

**理论上限**: 当前向量内核已接近 GPU 内存带宽上限（2184 MB/s），进一步优化空间有限（<30%）。

---

## 实测带宽分析 (2025-12-04)

### GPU 内存带宽测试
```
Host→GPU Upload:    4.5-6.6 GB/s
GPU→Host Download:  5.2-7.5 GB/s
GPU Internal Copy (scalar):  7.7-24 GB/s
GPU Internal Copy (vec16):   17-37 GB/s (峰值 4MB)
```

### LZO 解压缩带宽利用率
```
Scalar kernel:   1.2 GB/s
Vector kernel:   2.6 GB/s
Peak GPU BW:     37 GB/s
Utilization:     7% (远未饱和！)
```

**结论**: 当前性能瓶颈**不在带宽**，而在：
1. 解压缩算法的分支复杂度
2. 指令级并行度不足
3. 内存访问模式不规则

---

## 已实施优化 (循环展开)

### 优化内容
在 `offset >= 16` (无重叠) 场景下，添加 64 字节批次处理：
```c
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
```

### 性能提升
测试样本: `redis-video_parent_6_pages-1.img` (107 MB)
- **优化前**: 2184 MB/s (kernel)
- **优化后**: 2568 MB/s (kernel)
- **提升**: +17.6%

### 正确性验证
- ✅ 49/49 样本全部通过
- ✅ elasticsearch、influxdb、redis、nginx 全系列验证

---

## 下一步优化方向

基于带宽测试结果，优化重点应转向：

1. **减少分支**：合并 offset 判断逻辑
2. **提高指令并行度**：重排指令减少依赖
3. **优化中等重叠场景** (offset=9-15)：目前仍是标量
