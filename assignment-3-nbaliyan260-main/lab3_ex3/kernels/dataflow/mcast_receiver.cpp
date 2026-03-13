// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise 3: Batched multicast receiver kernel.
// Receives tiles_per_batch tiles per semaphore handshake.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int arg_idx = 0;
    uint32_t sender_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t n_tiles = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(0);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    uint64_t receivers_ready_sem_noc_addr = get_noc_addr(sender_x, sender_y, receivers_ready_semaphore_addr);

    uint32_t num_batches = n_tiles / tiles_per_batch;

    for (uint32_t batch = 0; batch < num_batches; batch++) {
        // Reserve space for entire batch
        cb_reserve_back(cb_id_in0, tiles_per_batch);

        noc_semaphore_set(tile_sent_sem_ptr, INVALID);

        // Signal sender that we're ready
        noc_semaphore_inc(receivers_ready_sem_noc_addr, 1);

        // Wait for sender to multicast the batch
        noc_semaphore_wait(tile_sent_sem_ptr, VALID);

        // Push entire batch to CB for compute kernel
        cb_push_back(cb_id_in0, tiles_per_batch);
    }
}
