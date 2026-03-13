// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lab 3, Exercise 4: Interior core reader (x, y) where x > 0 and y > 0.
// Receives both A and B slabs via multicast.
// A slab comes from the leftmost core in its row (0, y).
// B slab comes from the topmost core in its column (x, 0).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int arg_idx = 0;
    uint32_t M_block_tiles     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t N_block_tiles     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t K_block_tiles     = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_k_blocks      = get_arg_val<uint32_t>(arg_idx++);

    // A receive args
    uint32_t a_sender_x        = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_sender_y        = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_ready_sem_addr  = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t a_sent_sem_addr   = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    // B receive args
    uint32_t b_sender_x        = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_sender_y        = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_ready_sem_addr  = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t b_sent_sem_addr   = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    volatile tt_l1_ptr uint32_t* a_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_sent_sem_addr);
    volatile tt_l1_ptr uint32_t* b_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_sent_sem_addr);

    uint64_t a_ready_noc = get_noc_addr(a_sender_x, a_sender_y, a_ready_sem_addr);
    uint64_t b_ready_noc = get_noc_addr(b_sender_x, b_sender_y, b_ready_sem_addr);

    uint32_t A_slab_tiles = M_block_tiles * K_block_tiles;
    uint32_t B_slab_tiles = K_block_tiles * N_block_tiles;

    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        // ---- Receive A slab via multicast ----
        cb_reserve_back(cb_in0, A_slab_tiles);

        noc_semaphore_set(a_sent_ptr, INVALID);
        noc_semaphore_inc(a_ready_noc, 1);
        noc_semaphore_wait(a_sent_ptr, VALID);

        cb_push_back(cb_in0, A_slab_tiles);

        // ---- Receive B slab via multicast ----
        cb_reserve_back(cb_in1, B_slab_tiles);

        noc_semaphore_set(b_sent_ptr, INVALID);
        noc_semaphore_inc(b_ready_noc, 1);
        noc_semaphore_wait(b_sent_ptr, VALID);

        cb_push_back(cb_in1, B_slab_tiles);
    }
}
