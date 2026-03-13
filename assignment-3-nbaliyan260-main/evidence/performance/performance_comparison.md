# Performance Comparison (Assignment 3)

Profiling was done with **TT_METAL_DEVICE_PROFILER=1**, Release build, DPRINT and Watcher disabled. Device: Blackhole @ 1350 MHz.

## How to Reproduce

From the tt-metal repo root:

```bash
./build_metal.sh

# Exercise 2 (non-batched multicast)
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex2/lab3_ex2

# Exercise 3 (batched multicast, 10 tiles per batch)
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex3/lab3_ex3

# Exercise 4 (multi-core matmul with multicast, 5x5 grid)
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex4/lab3_ex4 5 5 2

# Lab 2 Ex2 baseline (multi-core matmul, no multicast, 5x5 grid)
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab2_matmul/lab2_matmul 5 5 2
```

After each run, device profiler data is written to `generated/profiler/.logs/profile_log_device.csv`. Firmware time is computed as (max ZONE_END time − min ZONE_START time) in cycles, then converted to ms using CHIP_FREQ (1350 MHz).

---

## Exercise 3: Batched vs Non-Batched Multicast

| Version              | Firmware time (cycles) | Firmware time (ms) |
|----------------------|------------------------|--------------------|
| Ex2 (1 tile/handshake)| ~389,000               | ~0.29              |
| Ex3 (10 tiles/batch)  | ~176,000               | ~0.13              |

**Conclusion:** Batched multicast (Ex3) is **noticeably faster** than per-tile handshake (Ex2), as expected from reduced semaphore synchronization rounds.

---

## Exercise 4: Multicast vs Lab 2 (No Multicast) MatMul

Same problem size and 5×5 core grid, K_block=2.

| Version                    | Firmware time (cycles) | Firmware time (ms) |
|----------------------------|------------------------|--------------------|
| Lab 2 Ex2 (no multicast)   | ~55,500                | ~0.041             |
| Ex4 (slab-level multicast) | ~29,500                | ~0.022             |

**Conclusion:** Multicast-enabled matmul (Ex4) **reduces firmware time** versus the Lab 2 data-reuse-only version by avoiding redundant DRAM reads; each A and B slab is read once per row/column and distributed via NoC.

---

## Profiler CSV Artifacts

Copies of the device profiler CSV used for the above comparisons are in the evidence subfolders:

- `evidence/exercise2/ex2_profile_log_device.csv` — Lab3 Ex2 (non-batched)
- `evidence/exercise3/ex3_profile_log_device.csv` — Lab3 Ex3 (batched)
- `evidence/exercise4/ex4_profile_log_device.csv` — Lab3 Ex4 (5×5 multicast matmul)
- `evidence/lab2_baseline/lab2_ex2_profile_log_device.csv` — Lab2 Ex2 (5×5 matmul, no multicast)

Exact cycle counts may vary slightly between runs; the comparisons above are from a single run per configuration.
