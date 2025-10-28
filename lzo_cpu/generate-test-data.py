#!/usr/bin/env python3
"""
LZ4 GPU Test Data Generator
生成具有重复模式的数据用于测试LZ4 GPU压缩解压的正确性
"""

import argparse
import os
import sys

def generate_repeated_pattern(size, pattern_size=64, pattern_density=0.7):
    """
    生成具有重复模式的数据

    Args:
        size: 总数据大小（字节）
        pattern_size: 重复模式大小（字节）
        pattern_density: 模式重复密度（0.0-1.0）
    """
    data = bytearray(size)

    # 生成基础重复模式
    pattern = bytearray()
    for i in range(pattern_size):
        if i % 4 == 0:
            pattern.append((i * 7 + 23) % 256)  # 伪随机模式
        elif i % 4 == 1:
            pattern.append(i % 256)
        elif i % 4 == 2:
            pattern.append((i * 13 + 47) % 256)
        else:
            pattern.append(255 - (i % 256))

    # 填充数据：部分区域使用重复模式，部分区域随机
    pos = 0
    while pos < size:
        # 决定这块区域使用模式还是随机
        if (pos // pattern_size) % 4 < int(pattern_density * 4):
            # 使用重复模式
            remaining = min(pattern_size, size - pos)
            for i in range(remaining):
                data[pos + i] = pattern[i % pattern_size]
        else:
            # 使用随机数据
            remaining = min(pattern_size * 2, size - pos)
            for i in range(remaining):
                data[pos + i] = (pos + i * 17 + 73) % 256

        pos += pattern_size * 2

    return bytes(data)

def generate_structured_data(size, structure_ratio=0.8):
    """
    生成具有结构化重复模式的数据

    Args:
        size: 总数据大小
        structure_ratio: 结构化数据比例
    """
    data = bytearray(size)
    struct_size = int(size * structure_ratio)

    # 前半部分：结构化重复数据
    pattern = b"0123456789ABCDEF" * 4  # 64字节模式
    for i in range(struct_size // len(pattern) + 1):
        start_pos = i * len(pattern)
        end_pos = min(start_pos + len(pattern), struct_size)
        if start_pos < struct_size:
            data[start_pos:end_pos] = pattern[:end_pos - start_pos]

    # 后半部分：随机数据
    for i in range(struct_size, size):
        data[i] = (i * 23 + 97) % 256

    return bytes(data)

def main():
    parser = argparse.ArgumentParser(description="生成LZ4 GPU测试数据")
    parser.add_argument("size", type=str, help="数据大小（如：64MB, 1GB）")
    parser.add_argument("--output", "-o", default="test_data.dat", help="输出文件名")
    parser.add_argument("--pattern", "-p", choices=["repeat", "structured", "mixed"],
                       default="repeat", help="数据模式类型")
    parser.add_argument("--pattern-size", type=int, default=64, help="重复模式大小")
    parser.add_argument("--density", type=float, default=0.7, help="模式密度")

    args = parser.parse_args()

    # 解析大小
    size_str = args.size.upper()
    if size_str.endswith('KB'):
        size = int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        size = int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        size = int(size_str[:-2]) * 1024 * 1024 * 1024
    elif size_str.endswith('B'):
        size = int(size_str[:-1])
    else:
        size = int(size_str)

    print(f"生成 {args.pattern} 模式测试数据，大小: {size:,} 字节")
    print(f"输出文件: {args.output}")

    # 根据模式生成数据
    if args.pattern == "repeat":
        data = generate_repeated_pattern(size, args.pattern_size, args.density)
    elif args.pattern == "structured":
        data = generate_structured_data(size)
    else:  # mixed
        # 混合模式：前半部分重复，后半部分结构化
        half_size = size // 2
        part1 = generate_repeated_pattern(half_size, args.pattern_size, args.density)
        part2 = generate_structured_data(half_size)
        data = part1 + part2

    # 写入文件
    with open(args.output, 'wb') as f:
        f.write(data)

    print(f"成功生成 {len(data):,} 字节测试数据")

    # 显示数据特征
    unique_bytes = len(set(data))
    print(f"唯一字节数: {unique_bytes}")
    print(f"数据复杂度: {unique_bytes/len(data)*100:.1f}%")

if __name__ == "__main__":
    main()