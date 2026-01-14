#!/bin/bash
# Power logger for CPU and GPU
# Usage: ./power_logger.sh <output_csv> <interval_sec> &
# PID=$!
# ... run work ...
# kill $PID

OUTPUT=$1
INTERVAL=$2

echo "timestamp,cpu_power_w,gpu_power_w" > "$OUTPUT"

while true; do
    TS=$(date +%s.%N)

    # Check for ENERGY_UJ files
    PKG_ENE="/sys/class/powercap/intel-rapl:0/energy_uj"
    GPU_ENE="/sys/class/powercap/intel-rapl:0:1/energy_uj" # uncore usually contains iGPU

    if [ -f "$PKG_ENE" ]; then
        E1_PKG=$(cat "$PKG_ENE")
        E1_GPU=0
        [ -f "$GPU_ENE" ] && E1_GPU=$(cat "$GPU_ENE")

        sleep "$INTERVAL"

        E2_PKG=$(cat "$PKG_ENE")
        E2_GPU=0
        [ -f "$GPU_ENE" ] && E2_GPU=$(cat "$GPU_ENE")

        # Calculate Delta Energy / Delta Time
        # Convert microjoules to Watts
        CPU_W=$(echo "scale=3; ($E2_PKG - $E1_PKG) / ($INTERVAL * 1000000)" | bc)
        GPU_W=$(echo "scale=3; ($E2_GPU - $E1_GPU) / ($INTERVAL * 1000000)" | bc)
    else
        CPU_W="0.0"
        GPU_W="0.0"
        sleep "$INTERVAL"
    fi

    echo "$TS,$CPU_W,$GPU_W" >> "$OUTPUT"
done
