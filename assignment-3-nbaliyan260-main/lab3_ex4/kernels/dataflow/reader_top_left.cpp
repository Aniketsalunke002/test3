// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lab 3, Exercise 4: Top-left core reader (0,0).
// Reads both A and B slabs from DRAM.
// Multicasts A slab across row 0, multicasts B slab down column 0.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int arg_idx = 0;
    uint32_t src0_addr         = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src1_addr         = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Kt                = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Nt                = get_arg_val<uint32_t>(arg_idx++);
    uint32_t block_row_start   = get_arg_val<uint32_t>(arg_idx++);
    uint32_t block_col_start   = get_arg_val<uint32_t>(arg_idx++);
    uint32_t M_block_tiles     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t N_block_tiles     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t K_block_tiles     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_k_blocks      = get_arg_val<uint32_t>(arg_idx++);

    uint32_t a_mcast_start_x   = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_mcast_start_y   = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_mcast_end_x     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_mcast_end_y     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_ready_sem_addr  = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t a_sent_sem_addr   = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t num_a_receivers   = get_arg_val<uint32_t>(arg_idx++);

    uint32_t b_mcast_start_x   = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_mcast_start_y   = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_mcast_end_x     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_mcast_end_y     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_ready_sem_addr  = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t b_sent_sem_addr   = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t num_b_receivers   = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    const uint32_t tile_size = get_tile_size(cb_in0);

    constexpr auto a_layout = TensorAccessorArgs<0>();
    const auto a_acc = TensorAccessor(a_layout, src0_addr, tile_size);
    constexpr auto b_layout = TensorAccessorArgs<1>();
    const auto b_acc = TensorAccessor(b_layout, src1_addr, tile_size);

    volatile tt_l1_ptr uint32_t* a_ready_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* a_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_sent_sem_addr);
    volatile tt_l1_ptr uint32_t* b_ready_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* b_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_sent_sem_addr);

    uint64_t a_sent_mcast = get_noc_multicast_addr(
        a_mcast_start_x, a_mcast_start_y, a_mcast_end_x, a_mcast_end_y, a_sent_sem_addr);
    uint64_t b_sent_mcast = get_noc_multicast_addr(
        b_mcast_start_x, b_mcast_start_y, b_mcast_end_x, b_mcast_end_y, b_sent_sem_addr);

    uint32_t A_slab_tiles = M_block_tiles * K_block_tiles;
    uint32_t B_slab_tiles = K_block_tiles * N_block_tiles;

    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        uint32_t k_start = kb * K_block_tiles;

        // ---- Read A slab from DRAM and multicast to row ----
        cb_reserve_back(cb_in0, A_slab_tiles);
        uint32_t a_start = get_write_ptr(cb_in0);
        uint32_t a_addr = a_start;

        for (uint32_t i = 0; i < M_block_tiles; ++i) {
            for (uint32_t k = 0; k < K_block_tiles; ++k) {
                uint32_t tid = (block_row_start + i) * Kt + k_start + k;
                noc_async_read_tile(tid, a_acc, a_addr);
                a_addr += tile_size;
            }
        }
        noc_async_read_barrier();

        noc_semaphore_wait(a_ready_ptr, num_a_receivers);
        noc_semaphore_set(a_ready_ptr, 0);

        uint64_t a_data_mcast = get_noc_multicast_addr(
            a_mcast_start_x, a_mcast_start_y, a_mcast_end_x, a_mcast_end_y, a_start);
        noc_async_write_multicast(a_start, a_data_mcast, tile_size * A_slab_tiles, num_a_receivers);
        noc_async_writes_flushed();

        *a_sent_ptr = VALID;
        noc_semaphore_set_multicast(a_sent_sem_addr, a_sent_mcast, num_a_receivers);
        noc_async_write_barrier();

        cb_push_back(cb_in0, A_slab_tiles);

        // ---- Read B slab from DRAM and multicast down column ----
        cb_reserve_back(cb_in1, B_slab_tiles);
        uint32_t b_start = get_write_ptr(cb_in1);
        uint32_t b_addr = b_start;

        for (uint32_t k = 0; k < K_block_tiles; ++k) {
            for (uint32_t j = 0; j < N_block_tiles; ++j) {
                uint32_t tid = (k_start + k) * Nt + block_col_start + j;
                noc_async_read_tile(tid, b_acc, b_addr);
                b_addr += tile_size;
            }
        }
        noc_async_read_barrier();

        noc_semaphore_wait(b_ready_ptr, num_b_receivers);
        noc_semaphore_set(b_ready_ptr, 0);

        uint64_t b_data_mcast = get_noc_multicast_addr(
            b_mcast_start_x, b_mcast_start_y, b_mcast_end_x, b_mcast_end_y, b_start);
        noc_async_write_multicast(b_start, b_data_mcast, tile_size * B_slab_tiles, num_b_receivers);
        noc_async_writes_flushed();

        *b_sent_ptr = VALID;
        noc_semaphore_set_multicast(b_sent_sem_addr, b_sent_mcast, num_b_receivers);
        noc_async_write_barrier();

        cb_push_back(cb_in1, B_slab_tiles);
    }
}
