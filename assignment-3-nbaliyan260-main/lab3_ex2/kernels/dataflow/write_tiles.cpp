// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int arg_idx = 0;
    uint32_t dst_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t n_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_idx = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr auto dst_layout_args = TensorAccessorArgs<0>();
    const auto dst_addr_gen = TensorAccessor(dst_layout_args, dst_base_addr, tile_size_bytes);

    uint32_t tile_offset = receiver_idx * n_tiles;

    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        uint32_t dram_tile_id = tile_offset + tile_idx;
        noc_async_write_tile(dram_tile_id, dst_addr_gen, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, 1);
    }
}
