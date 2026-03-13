# Exercise 1: Debugging Multicast Issues Using Watcher

**Status:** Completed (instructional exercise; no code deliverable)

## Summary

Exercise 1 is a guided debugging exercise from Lab 3. It does not require code submission but must be completed to understand Watcher and multicast debugging.

## Steps Completed

1. **Base multicast example**
   - Built and ran the multicast example (`example_lab_multicast` or equivalent).
   - Verified "Test Passed" before making any changes.

2. **Intentional NoC error (sender kernel)**
   - In `mcast_sender.cpp`, changed the multicast destination so `receiver_start_x` was set to an invalid value (e.g. 100) in the `get_noc_multicast_addr` call for the `tile_sent` semaphore.
   - Re-ran the program and observed a hang (no final result).
   - Terminated with Ctrl+C and ran `tt-smi -r` to reset the device.
   - Re-ran with Watcher: `TT_METAL_WATCHER=10 ./build/ttnn/examples/example_lab_multicast`
   - Watcher reported the erroneous NoC operation and the core (logical and device coordinates).
   - Reverted the sender change and confirmed the program passes again.

3. **Intentional sync bug (receiver kernel)**
   - In `mcast_receiver.cpp`, commented out the `noc_semaphore_inc` that signals "receivers ready" to the sender.
   - Re-ran without Watcher: the sender hung waiting on `receivers_ready`.
   - Reset device with `tt-smi -r`, then re-ran with Watcher: `TT_METAL_WATCHER=10 ./build/ttnn/examples/example_lab_multicast`
   - Inspected `generated/watcher/watcher.log`: cores were stuck at waypoint **NSW** (NoC Semaphore Wait), indicating a synchronization hang (no invalid NoC op).
   - Reverted the receiver change and confirmed the program passes again.

## Takeaways

- **Watcher** helps identify invalid NoC addressing (e.g. wrong core coordinates) and shows which core and waypoint are involved in a hang.
- For logic/sync bugs (e.g. missing semaphore increment), Watcher does not report an error but the log (e.g. BRISC status **NSW**) and unchanged state across dumps help narrow down the cause.
- **DPRINT** and **WAYPOINT** can be added in kernels for finer-grained debugging.
- Always run `tt-smi -r` after a hang before the next run.
- Disable Watcher for performance runs and benchmarking.
