#!/usr/bin/env bash
# Run all four programs with device profiler for performance comparison.
# Usage: from assignment repo root:  bash scripts/run_profiler_comparison.sh <tt-metal-root>
# Example: bash scripts/run_profiler_comparison.sh ../tt-metal
# Ensure DPRINT and Watcher are disabled; build with Release first: cd <tt-metal-root> && ./build_metal.sh
set -e
TT_ROOT="${1:?Usage: bash scripts/run_profiler_comparison.sh <tt-metal-root>}"
cd "$TT_ROOT"
export TT_METAL_DEVICE_PROFILER=1
echo "=== Lab3 Ex2 (non-batched multicast) ==="
./build/ttnn/examples/lab3_ex2/lab3_ex2
echo ""
echo "=== Lab3 Ex3 (batched multicast) ==="
./build/ttnn/examples/lab3_ex3/lab3_ex3
echo ""
echo "=== Lab3 Ex4 (5x5 multicast matmul) ==="
./build/ttnn/examples/lab3_ex4/lab3_ex4 5 5 2
echo ""
echo "=== Lab2 Ex2 (5x5 matmul, no multicast) ==="
./build/ttnn/examples/lab2_matmul/lab2_matmul 5 5 2
echo ""
echo "Done. Profiler CSV (last run) at: $TT_ROOT/generated/profiler/.logs/profile_log_device.csv"
echo "Copy to evidence/ after each run if you want to keep all four; see evidence/performance_comparison.md for interpretation."
