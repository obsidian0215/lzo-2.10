#!/usr/bin/env bash
set -euo pipefail

# script to precompile core × frontend kernel combinations
cd "$(dirname "$0")/.."

CORES=(lzo1x_1*.cl)
# if glob didn't match, keep literal pattern; check first element
if [ ! -e "${CORES[0]}" ]; then
    echo "No core kernel sources found matching lzo1x_1*.cl, skipping precompile-combos"
    exit 0
fi

FRONTS=(
    lzo1x_comp.cl
)

# ensure certain canonical cores and decompressors are always precompiled
CORE_ALWAYS=(
    lzo1x_1.cl
    lzo1x_1k.cl
    lzo1x_1l.cl
    lzo1x_1o.cl
    lzo1x_decomp.cl
    lzo1x_decomp_vec.cl
)

# Build canonical core/decompress binaries first (if present)
for s in "${CORE_ALWAYS[@]}"; do
    if [ -f "$s" ]; then
        bn=${s%%.cl}.bin
        echo "Building core/decomp $s -> $bn"
        ./build_kernel "$s" "$bn"
    fi
done

for c in "${CORES[@]}"; do
    cbase="${c%%.cl}"
    for f in "${FRONTS[@]}"; do
        if [ ! -f "$f" ]; then
            echo "Skipping missing frontend $f"
            continue
        fi
        fbase="${f%%.cl}"
        # strip leading 'lzo1x_' from frontend basename to avoid duplicate prefix
        fbase_short="${fbase#lzo1x_}"
        tmpdir=$(mktemp -d)
        cp "$f" "$tmpdir/$f"
        cp "$c" "$tmpdir/lzo1x_1.cl"
        # Always provide the canonical minilzo.h into the temp dir so core + frontend
        # have the necessary typedefs/macros. The header uses include-guards so
        # duplication is safe if frontend contains inline guards.
        if [ -f minilzo.h ]; then cp minilzo.h "$tmpdir/minilzo.h"; fi
        echo "Building combo $cbase + $fbase_short -> ${cbase}_${fbase_short}.bin"
        ./build_kernel "$tmpdir/$f" "${cbase}_${fbase_short}.bin"
        rm -rf "$tmpdir"
    done
done
