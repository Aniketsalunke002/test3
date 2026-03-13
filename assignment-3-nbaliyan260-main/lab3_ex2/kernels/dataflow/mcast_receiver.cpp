// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int arg_idx = 0;
    uint32_t sender_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t n_tiles = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    uint64_t receivers_ready_sem_noc_addr = get_noc_addr(sender_x, sender_y, receivers_ready_semaphore_addr);

    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        cb_reserve_back(cb_id_in0, 1);

        noc_semaphore_set(tile_sent_sem_ptr, INVALID);
        noc_semaphore_inc(receivers_ready_sem_noc_addr, 1);

        noc_semaphore_wait(tile_sent_sem_ptr, VALID);

        cb_push_back(cb_id_in0, 1);
    }
}
