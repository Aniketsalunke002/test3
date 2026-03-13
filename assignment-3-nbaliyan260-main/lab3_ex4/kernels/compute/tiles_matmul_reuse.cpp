#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

void kernel_main() {
 
    const uint32_t M_block_tiles = get_arg_val<uint32_t>(0);
    const uint32_t N_block_tiles = get_arg_val<uint32_t>(1);
    const uint32_t K_block_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_k_blocks = get_arg_val<uint32_t>(3);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;
    constexpr tt::CBIndex cb_partial = tt::CBIndex::c_24;

    const uint32_t a_slab_tiles = M_block_tiles * K_block_tiles;
    const uint32_t b_slab_tiles = K_block_tiles * N_block_tiles;

    // Initialize matmul engine once at kernel start.
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        // Ensure slabs for this K-block are in CBs.
        cb_wait_front(cb_in0, a_slab_tiles);
        cb_wait_front(cb_in1, b_slab_tiles);

        for (uint32_t i = 0; i < M_block_tiles; ++i) {
            for (uint32_t j = 0; j < N_block_tiles; ++j) {
                tile_regs_acquire();

                // If this is not the first K-block, load the existing partial sum tile into dst reg 0.
                if (kb != 0) {
                    cb_wait_front(cb_partial, 1);
                    copy_tile_to_dst_init_short(cb_partial);
                    copy_tile(cb_partial, 0, 0);
                    cb_pop_front(cb_partial, 1);

                    // Re-init matmul after copy.
                    mm_init_short(cb_in0, cb_in1);
                }

                // Accumulate contributions from this K-block.
                for (uint32_t k_local = 0; k_local < K_block_tiles; ++k_local) {
                    const uint32_t a_tile_index = i * K_block_tiles + k_local;          // row-major A_slab
                    const uint32_t b_tile_index = k_local * N_block_tiles + j;          // row-major B_slab
                    matmul_tiles(cb_in0, cb_in1, a_tile_index, b_tile_index, 0);
                }

                tile_regs_commit();
                tile_regs_wait();

                // Store updated result for this (i,j).
                if (kb == num_k_blocks - 1) {
                    // Last K-block: write final tile to output CB.
                    cb_reserve_back(cb_out, 1);
                    pack_tile(0, cb_out);
                    cb_push_back(cb_out, 1);
                } else {
                    // Not last: write partial tile back to intermediate CB.
                    cb_reserve_back(cb_partial, 1);
                    pack_tile(0, cb_partial);
                    cb_push_back(cb_partial, 1);
                }

                tile_regs_release();
            }
        }

        // Done using this slab, release input CB space.
        cb_pop_front(cb_in0, a_slab_tiles);
        cb_pop_front(cb_in1, b_slab_tiles);
    }
}
