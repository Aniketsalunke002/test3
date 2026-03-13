# Commands to Run Manually (with screenshot points)

**tt-metal root on this machine:**  
`/home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal`

---

## Step 0: Go to tt-metal and build (once)

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
./build_metal.sh
```

**SCREENSHOT 1:** When build finishes without errors (last lines of output).  
Save as `evidence/exercise1/01_build_success.png`

---

## Step 1: Exercise 2

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
./build/ttnn/examples/lab3_ex2/lab3_ex2
```

**SCREENSHOT 2:** When you see `[PASS] All 4 cores produced correct output` and **Test Passed**.  
Save as `evidence/exercise2/02_exercise2_passed.png`

---

## Step 2: Exercise 3

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
./build/ttnn/examples/lab3_ex3/lab3_ex3
```

**SCREENSHOT 3:** When you see `[PASS] All 4 cores produced correct output` and **Test Passed**.  
Save as `evidence/exercise3/03_exercise3_passed.png`

---

## Step 3: Exercise 4 (default 5×5)

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
./build/ttnn/examples/lab3_ex4/lab3_ex4 5 5 2
```

**SCREENSHOT 4:** When you see **Verification passed** and **Test Passed!**.  
Save as `evidence/exercise4/04_exercise4_passed.png`

---

## Step 4: Exercise 4 — extra grid (optional)

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
./build/ttnn/examples/lab3_ex4/lab3_ex4 2 2 2
```

**SCREENSHOT 5 (optional):** When you see **Verification passed** and **Test Passed!**.  
Save as `evidence/exercise4/05_exercise4_2x2_passed.png`

---

## Step 5: Lab 2 Ex2 (baseline, no multicast)

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
./build/ttnn/examples/lab2_matmul/lab2_matmul 5 5 2
```

**SCREENSHOT 6:** When you see **Verification passed**.  
Save as `evidence/lab2_baseline/06_lab2_ex2_passed.png`

---

## Step 6: Profiler — Exercise 2

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex2/lab3_ex2
```

**SCREENSHOT 7:** When you see **Test Passed** and “Profiler started on device 0” (or “Closing user mode device drivers”).  
Save as `evidence/exercise2/07_profiler_ex2.png`

---

## Step 7: Profiler — Exercise 3

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex3/lab3_ex3
```

**SCREENSHOT 8:** When you see **Test Passed**.  
Save as `evidence/exercise3/08_profiler_ex3.png`

---

## Step 8: Profiler — Exercise 4

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab3_ex4/lab3_ex4 5 5 2
```

**SCREENSHOT 9:** When you see **Verification passed** and **Test Passed!**.  
Save as `evidence/exercise4/09_profiler_ex4.png`

---

## Step 9: Profiler — Lab 2 Ex2

```bash
cd /home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal
TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/lab2_matmul/lab2_matmul 5 5 2
```

**SCREENSHOT 10:** When you see **Verification passed**.  
Save as `evidence/lab2_baseline/10_profiler_lab2.png`

---

## Summary

| Step | What you run              | Screenshot when you see                    |
|------|---------------------------|--------------------------------------------|
| 0    | Build                     | Build finished successfully                 |
| 1    | Ex2                       | Test Passed (4 cores)                      |
| 2    | Ex3                       | Test Passed (batched)                      |
| 3    | Ex4 5×5                   | Verification passed, Test Passed!          |
| 4    | Ex4 2×2 (optional)       | Verification passed, Test Passed!          |
| 5    | Lab2 Ex2 5×5              | Verification passed                        |
| 6–9  | Same runs with profiler   | Same as above + profiler ran               |

After taking screenshots, save them in the correct `evidence/<exercise>/` subfolder (see table in `evidence/README.md`), then commit and push.

**Path used:** `/home/nazishbaliyan/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal`
