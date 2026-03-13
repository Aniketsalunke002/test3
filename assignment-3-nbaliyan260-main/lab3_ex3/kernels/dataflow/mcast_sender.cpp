// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise 3: Batched multicast sender kernel.
// Reads tiles_per_batch tiles from DRAM, multicasts the entire batch at once,
// reducing semaphore handshake overhead.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int arg_idx = 0;
    uint32_t receiver_start_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_start_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_end_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_end_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t src_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t n_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_receivers = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(0);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr auto src_layout_args = TensorAccessorArgs<1>();
    const auto src_addr_gen = TensorAccessor(src_layout_args, src_base_addr, tile_size_bytes);

    volatile tt_l1_ptr uint32_t* receivers_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    uint64_t tile_sent_mcast_addr = get_noc_multicast_addr(
        receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, tile_sent_semaphore_addr);

    uint32_t num_batches = n_tiles / tiles_per_batch;
    uint32_t tile_idx = 0;

    for (uint32_t batch = 0; batch < num_batches; batch++) {
        cb_reserve_back(cb_id_in0, tiles_per_batch);
        uint32_t batch_start_addr = get_write_ptr(cb_id_in0);
        uint32_t cb_write_addr = batch_start_addr;

        // Read all tiles in the batch from DRAM
        for (uint32_t t = 0; t < tiles_per_batch; t++) {
            noc_async_read_tile(tile_idx, src_addr_gen, cb_write_addr);
            cb_write_addr += tile_size_bytes;
            tile_idx++;
        }
        // Single barrier after all reads in the batch
        noc_async_read_barrier();

        // Wait for all receivers to be ready
        noc_semaphore_wait(receivers_ready_sem_ptr, num_receivers);
        noc_semaphore_set(receivers_ready_sem_ptr, 0);

        // Multicast entire batch at once
        uint64_t batch_mcast_addr = get_noc_multicast_addr(
            receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, batch_start_addr);
        noc_async_write_multicast(batch_start_addr, batch_mcast_addr,
            tile_size_bytes * tiles_per_batch, num_receivers);

        noc_async_writes_flushed();

        // Signal receivers that batch has been sent
        *tile_sent_sem_ptr = VALID;
        noc_semaphore_set_multicast(tile_sent_semaphore_addr, tile_sent_mcast_addr, num_receivers);

        noc_async_write_barrier();

        // Push batch to local CB for compute kernel
        cb_push_back(cb_id_in0, tiles_per_batch);
    }
}
