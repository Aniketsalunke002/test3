# Assignment 3: Multicast for Improved Data Reuse in Multi-Core Matrix Multiplication

## Overview

This repository contains solutions for Lab 3 exercises implementing NoC multicast operations on Tenstorrent hardware. The exercises progressively build from a basic multicast example to a full multi-core matrix multiplication with cross-core data reuse via multicast.

**Matrix dimensions used:** M=640, K=320, N=640 (same as Lab 2)

---

## Prerequisites

- Tenstorrent hardware (Blackhole)
- tt-metal repository cloned and built
- Python 3.10+

## Setup Instructions

1. Copy each exercise directory (`lab3_ex2/`, `lab3_ex3/`, `lab3_ex4/`) into your tt-metal repository under `ttnn/examples/`.

2. Add the following to `ttnn/examples/CMakeLists.txt`:
   ```cmake
   add_subdirectory(lab3_ex2)
   add_subdirectory(lab3_ex3)
   add_subdirectory(lab3_ex4)
   ```

3. Build from the tt-metal root:
   ```bash
   ./build_metal.sh
   ```

---

## Exercise 2: Extending the Standalone Multicast Example

**Directory:** `lab3_ex2/`

**Description:** Extends the base multicast example so that the sender core also participates in computation and writeback. The output is 4 copies of the input tensor (1 sender + 3 receivers), verifying that the sender core correctly processes its own data alongside multicasting to receivers.

**Key changes from base multicast:**
- Compute and writer kernels created on ALL cores (including sender)
- Sender kernel modified: no `cb_wait_front`/`cb_pop_front` (compute kernel handles those), uses `cb_write_addr` for multicast source/destination, multicasts after DRAM read barrier, then pushes to local CB
- Sender core writes output at index 0, receivers at indices 1-3

**Files:**
- `lab3_ex2.cpp` - Host program
- `kernels/dataflow/mcast_sender.cpp` - Modified sender (reads DRAM + multicasts + feeds local compute)
- `kernels/dataflow/mcast_receiver.cpp` - Receiver (unchanged from base)
- `kernels/dataflow/write_tiles.cpp` - Writer (runs on all cores)
- `kernels/compute/tiles_copy.cpp` - Compute (runs on all cores)

**Build and Run:**
```bash
cd <tt-metal-root>
./build_metal.sh
./build/ttnn/examples/lab3_ex2/lab3_ex2
```

**Expected output:** All 4 cores (sender + 3 receivers) produce correct output. "Test Passed" message.

---

## Exercise 3: Batched Multicast for Improved Throughput

**Directory:** `lab3_ex3/`

**Description:** Reduces semaphore handshake overhead by transferring 10 tiles per batch instead of 1 tile at a time. Only one semaphore handshake is performed per batch, amortizing synchronization cost.

**Key changes from Exercise 2:**
- `TILES_PER_BATCH = 10` compile-time constant
- Input CB sized to `2 * TILES_PER_BATCH` tiles (double-buffered batches)
- Sender reads 10 tiles from DRAM in inner loop, single `noc_async_read_barrier` after all reads, then multicasts entire batch
- Receiver reserves 10 tiles at once, single semaphore handshake per batch
- Compute and writer kernels unchanged (still process tiles individually)

**Files:**
- `lab3_ex3.cpp` - Host program with batched CB configuration
- `kernels/dataflow/mcast_sender.cpp` - Batched sender
- `kernels/dataflow/mcast_receiver.cpp` - Batched receiver
- `kernels/dataflow/write_tiles.cpp` - Writer (unchanged)
- `kernels/compute/tiles_copy.cpp` - Compute (unchanged)

**Build and Run:**
```bash
cd <tt-metal-root>
./build_metal.sh
./build/ttnn/examples/lab3_ex3/lab3_ex3
```

**Expected output:** All 4 cores produce correct output. "Test Passed" message.

---

## Exercise 4: Multi-Core Matrix Multiplication with Multicast

**Directory:** `lab3_ex4/`

**Description:** Applies slab-level multicast to the blocked multi-core matrix multiplication from Lab 2 Exercise 2. Instead of every core reading A and B tiles from DRAM, only edge cores read from DRAM and multicast to their row/column:
- **Left-column cores** read A slabs from DRAM and multicast across their row
- **Top-row cores** read B slabs from DRAM and multicast down their column
- **Interior cores** receive both A and B slabs via multicast (zero DRAM reads)
- **Top-left core** reads both A and B and multicasts both

This reduces redundant DRAM traffic proportionally to the grid size.

**Core roles (Figure 5 from Lab 3):**
```
  (0,0) Purple  | (1,0) Blue   | (2,0) Blue   | ...
  Top-Left      | Top-Row      | Top-Row       |
  Sends A & B   | Recv A,Send B| Recv A,Send B |
  ──────────────┼──────────────┼───────────────┤
  (0,1) Red     | (1,1) White  | (2,1) White   | ...
  Left-Col      | Interior     | Interior      |
  Send A,Recv B | Recv A & B   | Recv A & B    |
```

**Semaphores (4 total):**
1. `receivers_ready_A` - A receivers signal readiness to row sender
2. `slab_sent_A` - A sender signals slab multicast complete
3. `receivers_ready_B` - B receivers signal readiness to column sender
4. `slab_sent_B` - B sender signals slab multicast complete

**Files:**
- `lab3_ex4.cpp` - Host program with 4 core roles, 4 semaphores, 4 reader kernels
- `kernels/dataflow/reader_top_left.cpp` - Reads A & B from DRAM, multicasts both
- `kernels/dataflow/reader_left_col.cpp` - Reads A from DRAM + multicasts; receives B
- `kernels/dataflow/reader_top_row.cpp` - Receives A; reads B from DRAM + multicasts
- `kernels/dataflow/reader_interior.cpp` - Receives both A and B via multicast
- `kernels/dataflow/write_tiles_reuse.cpp` - Writer kernel (from Lab 2)
- `kernels/compute/tiles_matmul_reuse.cpp` - Compute kernel (from Lab 2)

**Build and Run:**
```bash
cd <tt-metal-root>
./build_metal.sh

# Default: 5x5 grid, K_block=2
./build/ttnn/examples/lab3_ex4/lab3_ex4

# Custom grid and K_block:
./build/ttnn/examples/lab3_ex4/lab3_ex4 <grid_x> <grid_y> [K_block]

# Examples:
./build/ttnn/examples/lab3_ex4/lab3_ex4 2 2 2    # 2x2 grid
./build/ttnn/examples/lab3_ex4/lab3_ex4 4 5 2    # 4x5 grid
./build/ttnn/examples/lab3_ex4/lab3_ex4 5 5 2    # 5x5 grid
```

**Expected output:** "Verification passed" and "Test Passed!" for all grid configurations.

**Tested configurations (all pass):**
- 2x2 grid (K_block=2)
- 4x5 grid (K_block=2)
- 5x4 grid (K_block=2)
- 5x5 grid (K_block=2)

---

## Evidence

Evidence is organized by exercise under `evidence/`:

| Folder | Contents |
|--------|----------|
| **evidence/exercise1/** | Watcher debugging: `exercise1_completed.md`, `01_build_success.png` |
| **evidence/exercise2/** | Ex2: `exercise2_output.txt`, `ex2_profile_log_device.csv`, `02_exercise2_passed.png`, `07_profiler_ex2.png` |
| **evidence/exercise3/** | Ex3: `exercise3_output.txt`, `ex3_profile_log_device.csv`, `03_exercise3_passed.png`, `08_profiler_ex3.png` |
| **evidence/exercise4/** | Ex4: `exercise4_output.txt`, `ex4_profile_log_device.csv`, `04_exercise4_passed.png`, `05_exercise4_2x2_passed.png`, `09_profiler_ex4.png` |
| **evidence/lab2_baseline/** | Lab2 Ex2 (comparison): `lab2_ex2_profile_log_device.csv`, `06_lab2_ex2_passed.png`, `10_profiler_lab2.png` |
| **evidence/performance/** | `performance_comparison.md` (Ex2 vs Ex3, Lab2 vs Ex4) |

See `evidence/README.md` for a short description of each folder.

---

## Performance Comparison

Build with Release; run with device profiler enabled (DPRINT and Watcher **disabled**):

```bash
cd <tt-metal-root>
./build_metal.sh

# Exercise 2 (non-batched):
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex2/lab3_ex2

# Exercise 3 (batched):
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex3/lab3_ex3

# Exercise 4 (5×5 multicast matmul):
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex4/lab3_ex4 5 5 2

# Lab 2 Ex2 baseline (5×5, no multicast):
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab2_matmul/lab2_matmul 5 5 2
```

The profiler writes to `<tt-metal-root>/generated/profiler/.logs/profile_log_device.csv` (overwritten each run). Copies of the CSVs used for the comparison are in `evidence/exercise2/`, `evidence/exercise3/`, `evidence/exercise4/`, and `evidence/lab2_baseline/`.

**Optional — run all four in one go:** from this repo root, after building tt-metal:
```bash
bash scripts/run_profiler_comparison.sh /path/to/tt-metal
```

Results and interpretation are in `evidence/performance/performance_comparison.md`.

---

## Part 2: Cross-Validation (Peer Review)

When a classmate is assigned to you:

1. **Grant access** to your repo as instructed in class.
2. They will follow this README to build/run and reproduce results.
3. They will check outputs against the claims in this README and in `evidence/`.
4. They will open an **issue** on this repo with their evidence and the checklist below.
5. You may reply to that issue and to any issue you open on their repo.

---

## Assignment 3 Checklist

### Part 1: Your Lab
- [x] Completed **Exercise 1** (Watcher debugging; see `evidence/exercise1/exercise1_completed.md`)
- [x] Completed **Exercise 2** (sender core also computes; code in `lab3_ex2/`)
- [x] Completed **Exercise 3** (batched multicast; code in `lab3_ex3/`)
- [x] Performance comparison done for Exercise 3 (Ex2 vs Ex3 in `evidence/performance/performance_comparison.md`)
- [x] Completed **Exercise 4** (multicast multi-core matmul; code in `lab3_ex4/`)
- [x] Performance comparison done for Exercise 4 (Lab2 vs Ex4 in `evidence/performance/performance_comparison.md`)
- [x] Correctness verified (Test Passed / Verification passed for all exercises)
- [x] Clear build/run instructions in this README
- [x] All work committed and pushed to GitHub Classroom repo

### Part 2: Cross-Validation
- [ ] Read and followed the assigned classmate's README
- [ ] Reproduced and verified their results (or documented failures)
- [ ] Opened an issue on their repo with evidence and this checklist
- [ ] Documented and pushed your reproduction attempts to your own repo
