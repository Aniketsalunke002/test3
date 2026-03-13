#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t Nt = get_arg_val<uint32_t>(1);

    const uint32_t block_row_start = get_arg_val<uint32_t>(2);
    const uint32_t block_col_start = get_arg_val<uint32_t>(3);

    const uint32_t M_block_tiles = get_arg_val<uint32_t>(4);
    const uint32_t N_block_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    // Write C_block in row-major order.
    for (uint32_t i = 0; i < M_block_tiles; ++i) {
        const uint32_t out_row = block_row_start + i;
        for (uint32_t j = 0; j < N_block_tiles; ++j) {
            const uint32_t out_col = block_col_start + j;
            const uint32_t tile_id = out_row * Nt + out_col;

            cb_wait_front(cb_id_out, 1);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_tile(tile_id, c, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}
