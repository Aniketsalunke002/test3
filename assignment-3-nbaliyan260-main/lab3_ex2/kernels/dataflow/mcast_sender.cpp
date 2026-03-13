// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise 2: Modified sender kernel that also feeds local compute.
// Key change: no cb_wait_front/cb_pop_front (compute kernel handles those).
// Uses cb_write_addr for multicast source/destination instead of cb_read_addr.

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

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr auto src_layout_args = TensorAccessorArgs<0>();
    const auto src_addr_gen = TensorAccessor(src_layout_args, src_base_addr, tile_size_bytes);

    volatile tt_l1_ptr uint32_t* receivers_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    uint64_t tile_sent_mcast_addr = get_noc_multicast_addr(
        receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, tile_sent_semaphore_addr);

    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t cb_write_addr = get_write_ptr(cb_id_in0);

        noc_async_read_tile(tile_idx, src_addr_gen, cb_write_addr);
        noc_async_read_barrier();

        // Wait for all receivers to be ready
        noc_semaphore_wait(receivers_ready_sem_ptr, num_receivers);
        noc_semaphore_set(receivers_ready_sem_ptr, 0);

        // Multicast using cb_write_addr (same address on all receiver cores due to CB sync)
        uint64_t tile_mcast_addr =
            get_noc_multicast_addr(receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, cb_write_addr);
        noc_async_write_multicast(cb_write_addr, tile_mcast_addr, tile_size_bytes, num_receivers);

        noc_async_writes_flushed();

        // Signal receivers that tile has been sent
        *tile_sent_sem_ptr = VALID;
        noc_semaphore_set_multicast(tile_sent_semaphore_addr, tile_sent_mcast_addr, num_receivers);

        noc_async_write_barrier();

        // Push to local CB so compute kernel can consume the tile
        cb_push_back(cb_id_in0, 1);
    }
}
