#!/usr/bin/env bash
set -euo pipefail

# Script to precompile standalone kernel binaries
# Frontend combinations (comp/atomic) removed - use independent .cl files only
cd "$(dirname "$0")/.."

# List of all kernels to precompile
KERNELS=(
    lzo1x_1.cl
    lzo1x_1k.cl
    lzo1x_1l.cl
    lzo1x_1o.cl
    lzo1x_decomp.cl
    lzo1x_decomp_vec.cl
)

echo "Precompiling standalone kernels (no frontend combinations)..."

for kernel in "${KERNELS[@]}"; do
    if [ -f "$kernel" ]; then
        bin="${kernel%%.cl}.bin"
        echo "Building $kernel -> $bin"
        ./build_kernel "$kernel" "$bin"
    else
        echo "Warning: $kernel not found, skipping"
    fi
done

echo "Precompilation complete!"
