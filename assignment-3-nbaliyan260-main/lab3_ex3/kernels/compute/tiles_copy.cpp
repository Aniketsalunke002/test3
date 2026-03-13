// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t dst_reg_idx = 0;

    unary_op_init_common(cb_in0, cb_out0);
    copy_tile_init(cb_in0);

    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        cb_wait_front(cb_in0, 1);

        tile_regs_acquire();
        copy_tile(cb_in0, 0, dst_reg_idx);
        cb_pop_front(cb_in0, 1);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg_idx, cb_out0);
        cb_push_back(cb_out0, 1);

        tile_regs_release();
    }
}
