# Assignment 3 — Work Completed So Far

This file summarizes everything that has been done for Assignment 3 (Lab 3: Multicast for Improved Data Reuse in Multi-Core Matrix Multiplication).

---

## 1. Repository Structure

- **Assignment repo:** `assignment-3-nbaliyan260` (this repo)
- **Code locations:** Each exercise in its own folder: `lab3_ex2/`, `lab3_ex3/`, `lab3_ex4/`
- **Evidence:** All evidence in `evidence/`, organized by exercise in subfolders `exercise1/`, `exercise2/`, `exercise3/`, `exercise4/`, `lab2_baseline/`, `performance/`
- **Scripts:** Profiler helper in `scripts/run_profiler_comparison.sh`

---

## 2. What Was Done (By Deliverable)

### Exercise 1: Debugging Multicast Issues Using Watcher

- **Deliverable:** No code to submit; instructional only.
- **Done:**
  - Base multicast example built and run; "Test Passed" verified.
  - NoC error introduced in sender (invalid multicast destination) → program hung → Watcher used → error and core coordinates reported.
  - Sync bug introduced in receiver (commented `noc_semaphore_inc`) → sender hung → Watcher log showed NSW (NoC Semaphore Wait).
  - Changes reverted; program passes again.
- **Evidence:** `evidence/exercise1/exercise1_completed.md`, `evidence/exercise1/01_build_success.png`.

---

### Exercise 2: Extending the Standalone Multicast Example

- **Deliverable:** Sender core also does compute and writeback (same as receivers).
- **Done:**
  - Copied base multicast into `lab3_ex2/` (host + kernels).
  - Host: compute and writer kernels created on **all 4 cores** (sender + 3 receivers); runtime args and output tensor sized for 4 cores.
  - Sender kernel: no `cb_wait_front`/`cb_pop_front`; uses `cb_write_addr` for multicast source/destination; multicasts after DRAM read barrier, then pushes to local CB for compute.
  - Verification checks all 4 cores’ output.
- **Code:** `lab3_ex2/lab3_ex2.cpp`, `lab3_ex2/kernels/dataflow/mcast_sender.cpp`, mcast_receiver, write_tiles, compute tiles_copy.
- **Evidence:** `evidence/exercise2/` (exercise2_output.txt, ex2_profile_log_device.csv, screenshots).

---

### Exercise 3: Batched Multicast for Improved Throughput

- **Deliverable:** 10 tiles per semaphore handshake instead of 1.
- **Done:**
  - Copied Ex2 into `lab3_ex3/`.
  - Host: `TILES_PER_BATCH = 10`; assertion that total tiles divisible by 10; CBs sized for `2 * TILES_PER_BATCH` (double-buffered batches).
  - Sender kernel: reserves `tiles_per_batch`, reads all tiles in inner loop, single `noc_async_read_barrier` after batch, then one multicast + semaphore handshake per batch.
  - Receiver kernel: reserves `tiles_per_batch`, one semaphore handshake per batch, then push `tiles_per_batch`.
  - Compute and writer unchanged.
- **Code:** `lab3_ex3/lab3_ex3.cpp`, batched `mcast_sender.cpp`, batched `mcast_receiver.cpp`, same write_tiles and tiles_copy.
- **Evidence:** `evidence/exercise3/` (exercise3_output.txt, ex3_profile_log_device.csv, screenshots).
- **Performance:** Ex3 (batched) faster than Ex2 (per-tile); numbers in `evidence/performance/performance_comparison.md`.

---

### Exercise 4: Multi-Core Matrix Multiplication with Multicast

- **Deliverable:** Lab 2 Ex2 matmul extended with slab-level multicast for A (along rows) and B (along columns).
- **Done:**
  - New project `lab3_ex4/`; compute and writer kernels reused from Lab 2 Ex2.
  - Four core roles: **top-left** (reads A & B from DRAM, multicasts both), **left-column** (reads A, multicasts A; receives B), **top-row** (receives A; reads B, multicasts B), **interior** (receives both A and B).
  - Four semaphores: receivers_ready_A, slab_sent_A, receivers_ready_B, slab_sent_B (all created on all cores).
  - Four reader kernels: `reader_top_left.cpp`, `reader_left_col.cpp`, `reader_top_row.cpp`, `reader_interior.cpp`.
  - Slab-level multicast (full slab per handshake), not tile-by-tile.
  - Host: core ranges per role, `SetRuntimeArgs` per core with correct device coords and semaphore IDs.
- **Code:** `lab3_ex4/lab3_ex4.cpp`, four reader kernels in `lab3_ex4/kernels/dataflow/`, `write_tiles_reuse.cpp`, `tiles_matmul_reuse.cpp` from Lab 2.
- **Evidence:** `evidence/exercise4/` (exercise4_output.txt, ex4_profile_log_device.csv, screenshots); Lab2 baseline in `evidence/lab2_baseline/`.
- **Performance:** Ex4 (multicast) faster than Lab 2 Ex2 (no multicast); table in `evidence/performance/performance_comparison.md`.

---

## 3. Build and Integration

- **tt-metal:** Lab3 code lives under `ttnn/examples/` (lab3_ex2, lab3_ex3, lab3_ex4). Base `lab_multicast` was added from Tenstorrent repo; `CMakeLists.txt` updated to add these subdirectories.
- **Fixes applied:** Added `#include <tt-metalium/tensor_accessor_args.hpp>` where needed; compute kernel includes changed from `api/compute/` to `compute_kernel_api/` in tiles_copy (and equivalent) so builds succeed.
- **Assignment repo:** Contains copies of the **source** for lab3_ex2, lab3_ex3, lab3_ex4 (and README, evidence, scripts). To run, these directories are copied into the tt-metal tree and built with `./build_metal.sh`.

---

## 4. Evidence and Profiling

- **Test outputs:** `evidence/exercise2/`, `evidence/exercise3/`, `evidence/exercise4/` (logs and screenshots; all show "Test Passed" / "Verification passed").
- **Exercise 1:** `evidence/exercise1/`.
- **Performance:** `evidence/performance/performance_comparison.md` — Ex2 vs Ex3 (batched faster), Lab2 Ex2 vs Ex4 (multicast faster); firmware times in cycles/ms.
- **Profiler CSVs:** in `evidence/exercise2/`, `evidence/exercise3/`, `evidence/exercise4/`, `evidence/lab2_baseline/`.
- **Script:** `scripts/run_profiler_comparison.sh` — runs all four programs with profiler from tt-metal root.

---

## 5. README and Checklist

- **README.md:**  
  - Prerequisites and setup.  
  - Build/run for each exercise (Ex2, Ex3, Ex4).  
  - Evidence table (by folder: `evidence/exercise1/`, etc.).  
  - Performance comparison commands and note on profiler CSV path; optional use of `scripts/run_profiler_comparison.sh`.  
  - Part 2: Cross-validation (what to do when a classmate is assigned).  
  - Full Assignment 3 checklist (Part 1 all checked; Part 2 unchecked until peer is assigned).

---

## 6. Git and Submission

- All of the above is committed and pushed to the GitHub Classroom repo:  
  `https://github.com/MBZUAI-7201-8201-SP26/assignment-3-nbaliyan260`
- Part 1 of the assignment (your lab + evidence + performance) is complete; Part 2 (cross-validation) is pending assignment of a classmate.

---

## 7. Quick Reference — Where Things Are

| Item | Location |
|------|----------|
| Ex2 host + kernels | `lab3_ex2/` |
| Ex3 host + kernels | `lab3_ex3/` |
| Ex4 host + kernels | `lab3_ex4/` |
| Ex1 | `evidence/exercise1/` |
| Ex2/3/4 run logs & screenshots | `evidence/exercise2/`, `evidence/exercise3/`, `evidence/exercise4/` |
| Lab2 baseline | `evidence/lab2_baseline/` |
| Performance comparison | `evidence/performance/performance_comparison.md` |
| Profiler script | `scripts/run_profiler_comparison.sh` |
| Build/run and checklist | `README.md` |

---

*Last updated: reflects work completed for Assignment 3 (Lab 3) through the addition of evidence, performance comparison, README, and profiler script.*
