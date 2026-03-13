// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lab 3, Exercise 4: Multi-Core Matrix Multiplication with Multicast
// Extends Lab 2 Ex2 blocked matmul with slab-level multicast:
//   - Left-column cores read A from DRAM and multicast across their row
//   - Top-row cores read B from DRAM and multicast down their column
//   - Interior cores receive both A and B via multicast
//   - Top-left core reads both A and B from DRAM and multicasts both

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace ttnn;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static void create_cb(Program& program, const CoreRangeSet& cores, uint32_t num_tiles, tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    CircularBufferConfig cb_config =
        CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
            .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, cores, cb_config);
}

static std::vector<float> reference_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    uint32_t M,
    uint32_t K,
    uint32_t N) {
    std::vector<float> out(M * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<float>(a[i * K + k]) * static_cast<float>(b[k * N + j]);
            }
            out[i * N + j] = acc;
        }
    }
    return out;
}

static void verify_against_reference(
    const std::vector<float>& ref,
    const std::vector<bfloat16>& got,
    float rtol = 1e-1f,
    float atol = 1e-1f) {
    TT_FATAL(ref.size() == got.size(), "Size mismatch between reference and device output");

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float r = ref[i];
        const float g = static_cast<float>(got[i]);
        const float abs_err = std::fabs(r - g);
        const float rel_err = abs_err / (std::fabs(r) + 1e-6f);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);

        if (abs_err > atol && rel_err > rtol) {
            std::cout << "Mismatch at i=" << i << " ref=" << r << " got=" << g
                      << " abs_err=" << abs_err << " rel_err=" << rel_err << "\n";
            TT_FATAL(false, "Mismatch in output");
        }
    }
    std::cout << "Verification passed. max_abs_err=" << max_abs << " max_rel_err=" << max_rel << "\n";
}

static std::vector<bfloat16> matmul_multicast(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    CoreCoord core_grid,
    uint32_t K_block_tiles) {

    const CoreCoord available_grid = mesh_device->compute_with_storage_grid_size();
    TT_FATAL(core_grid.x >= 2 && core_grid.y >= 2, "Core grid must be at least 2x2 for multicast");
    TT_FATAL(core_grid.x <= available_grid.x && core_grid.y <= available_grid.y,
        "Requested core grid exceeds device compute grid");
    TT_FATAL(M % TILE_HEIGHT == 0 && N % TILE_WIDTH == 0 && K % TILE_WIDTH == 0,
        "Matrix dims must be tile-aligned");

    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Nt = N / TILE_WIDTH;
    const uint32_t Kt = K / TILE_WIDTH;

    TT_FATAL(K_block_tiles > 0, "K_block_tiles must be > 0");
    TT_FATAL(Kt % K_block_tiles == 0, "K_block_tiles must divide Kt evenly");
    TT_FATAL(Mt % core_grid.y == 0, "Mt must be divisible by core_grid.y");
    TT_FATAL(Nt % core_grid.x == 0, "Nt must be divisible by core_grid.x");

    const uint32_t M_block_tiles = Mt / core_grid.y;
    const uint32_t N_block_tiles = Nt / core_grid.x;
    const uint32_t num_k_blocks = Kt / K_block_tiles;

    const uint32_t A_slab_tiles = M_block_tiles * K_block_tiles;
    const uint32_t B_slab_tiles = K_block_tiles * N_block_tiles;
    const uint32_t C_block_tiles = M_block_tiles * N_block_tiles;

    const uint32_t num_a_receivers = core_grid.x - 1;
    const uint32_t num_b_receivers = core_grid.y - 1;

    std::cout << "[Ex4-Multicast] core_grid=" << core_grid.x << "x" << core_grid.y
              << "  M_block=" << M_block_tiles << "  N_block=" << N_block_tiles
              << "  K_block=" << K_block_tiles << "  num_k_blocks=" << num_k_blocks
              << "  A_receivers=" << num_a_receivers << "  B_receivers=" << num_b_receivers << "\n";

    // ---- Core ranges for the four roles ----
    CoreRange all_cores_logical(CoreCoord(0, 0), CoreCoord(core_grid.x - 1, core_grid.y - 1));

    CoreRange top_left_range(CoreCoord(0, 0), CoreCoord(0, 0));
    CoreRange left_col_range(CoreCoord(0, 1), CoreCoord(0, core_grid.y - 1));
    CoreRange top_row_range(CoreCoord(1, 0), CoreCoord(core_grid.x - 1, 0));
    CoreRange interior_range(CoreCoord(1, 1), CoreCoord(core_grid.x - 1, core_grid.y - 1));

    CoreRangeSet all_cores_set{all_cores_logical};

    // ---- Device tensors ----
    const TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));

    Tensor src0_tensor = Tensor::from_vector<bfloat16>(a, TensorSpec(Shape({M, K}), tile_layout), mesh_device.get());
    Tensor src1_tensor = Tensor::from_vector<bfloat16>(b, TensorSpec(Shape({K, N}), tile_layout), mesh_device.get());
    Tensor dst_tensor = create_device_tensor(TensorSpec(Shape({M, N}), tile_layout), mesh_device.get());

    auto src0_buf = src0_tensor.mesh_buffer();
    auto src1_buf = src1_tensor.mesh_buffer();
    auto dst_buf = dst_tensor.mesh_buffer();
    TT_FATAL(src0_buf && src1_buf && dst_buf, "Failed to get mesh buffers");

    // ---- Program ----
    Program program = CreateProgram();

    // CBs: double-buffered slabs
    create_cb(program, all_cores_set, 2 * A_slab_tiles, tt::CBIndex::c_0);
    create_cb(program, all_cores_set, 2 * B_slab_tiles, tt::CBIndex::c_1);
    create_cb(program, all_cores_set, C_block_tiles, tt::CBIndex::c_16);
    create_cb(program, all_cores_set, C_block_tiles, tt::CBIndex::c_24);

    // ---- Semaphores (4 total, created on all cores) ----
    uint32_t receivers_ready_A = CreateSemaphore(program, all_cores_logical, 0);
    uint32_t slab_sent_A       = CreateSemaphore(program, all_cores_logical, INVALID);
    uint32_t receivers_ready_B = CreateSemaphore(program, all_cores_logical, 0);
    uint32_t slab_sent_B       = CreateSemaphore(program, all_cores_logical, INVALID);

    // ---- Reader kernels (4 role-specific) ----

    // Top-left: reads A and B from DRAM
    std::vector<uint32_t> tl_compile_args;
    TensorAccessorArgs(*src0_buf).append_to(tl_compile_args);
    TensorAccessorArgs(*src1_buf).append_to(tl_compile_args);

    KernelHandle reader_top_left_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex4/kernels/dataflow/reader_top_left.cpp",
        top_left_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = tl_compile_args});

    // Left column: reads A from DRAM
    std::vector<uint32_t> lc_compile_args;
    TensorAccessorArgs(*src0_buf).append_to(lc_compile_args);

    KernelHandle reader_left_col_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex4/kernels/dataflow/reader_left_col.cpp",
        left_col_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = lc_compile_args});

    // Top row: reads B from DRAM
    std::vector<uint32_t> tr_compile_args;
    TensorAccessorArgs(*src1_buf).append_to(tr_compile_args);

    KernelHandle reader_top_row_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex4/kernels/dataflow/reader_top_row.cpp",
        top_row_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = tr_compile_args});

    // Interior: receives both via multicast (no TensorAccessorArgs needed)
    KernelHandle reader_interior_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex4/kernels/dataflow/reader_interior.cpp",
        interior_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});

    // ---- Writer kernel (same for all cores, from Lab 2) ----
    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(*dst_buf).append_to(writer_compile_args);

    KernelHandle writer_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex4/kernels/dataflow/write_tiles_reuse.cpp",
        all_cores_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default,
                           .compile_args = writer_compile_args});

    // ---- Compute kernel (same for all cores, from Lab 2) ----
    KernelHandle compute_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex4/kernels/compute/tiles_matmul_reuse.cpp",
        all_cores_set,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // ---- Per-core runtime arguments ----
    for (uint32_t y = 0; y < core_grid.y; ++y) {
        for (uint32_t x = 0; x < core_grid.x; ++x) {
            const CoreCoord core_logical{x, y};

            const uint32_t block_row_start = y * M_block_tiles;
            const uint32_t block_col_start = x * N_block_tiles;

            // Compute and writer args are the same for all roles
            SetRuntimeArgs(program, compute_id, core_logical,
                {M_block_tiles, N_block_tiles, K_block_tiles, num_k_blocks});
            SetRuntimeArgs(program, writer_id, core_logical,
                {dst_buf->address(), Nt, block_row_start, block_col_start, M_block_tiles, N_block_tiles});

            if (x == 0 && y == 0) {
                // Top-left core: sends A across row 0, sends B down column 0
                CoreCoord a_recv_start_device = mesh_device->worker_core_from_logical_core({1, 0});
                CoreCoord a_recv_end_device = mesh_device->worker_core_from_logical_core({core_grid.x - 1, 0});
                CoreCoord b_recv_start_device = mesh_device->worker_core_from_logical_core({0, 1});
                CoreCoord b_recv_end_device = mesh_device->worker_core_from_logical_core({0, core_grid.y - 1});

                SetRuntimeArgs(program, reader_top_left_id, core_logical,
                    {src0_buf->address(), src1_buf->address(),
                     Kt, Nt, block_row_start, block_col_start,
                     M_block_tiles, N_block_tiles, K_block_tiles, num_k_blocks,
                     // A multicast
                     static_cast<uint32_t>(a_recv_start_device.x),
                     static_cast<uint32_t>(a_recv_start_device.y),
                     static_cast<uint32_t>(a_recv_end_device.x),
                     static_cast<uint32_t>(a_recv_end_device.y),
                     receivers_ready_A, slab_sent_A, num_a_receivers,
                     // B multicast
                     static_cast<uint32_t>(b_recv_start_device.x),
                     static_cast<uint32_t>(b_recv_start_device.y),
                     static_cast<uint32_t>(b_recv_end_device.x),
                     static_cast<uint32_t>(b_recv_end_device.y),
                     receivers_ready_B, slab_sent_B, num_b_receivers});

            } else if (x == 0 && y > 0) {
                // Left column core: sends A across row y, receives B from (0, 0)
                CoreCoord a_recv_start_device = mesh_device->worker_core_from_logical_core({1, y});
                CoreCoord a_recv_end_device = mesh_device->worker_core_from_logical_core({core_grid.x - 1, y});
                CoreCoord b_sender_device = mesh_device->worker_core_from_logical_core({0, 0});

                SetRuntimeArgs(program, reader_left_col_id, core_logical,
                    {src0_buf->address(), Kt, block_row_start,
                     M_block_tiles, N_block_tiles, K_block_tiles, num_k_blocks,
                     // A multicast
                     static_cast<uint32_t>(a_recv_start_device.x),
                     static_cast<uint32_t>(a_recv_start_device.y),
                     static_cast<uint32_t>(a_recv_end_device.x),
                     static_cast<uint32_t>(a_recv_end_device.y),
                     receivers_ready_A, slab_sent_A, num_a_receivers,
                     // B receive
                     static_cast<uint32_t>(b_sender_device.x),
                     static_cast<uint32_t>(b_sender_device.y),
                     receivers_ready_B, slab_sent_B});

            } else if (x > 0 && y == 0) {
                // Top row core: receives A from (0, 0), sends B down column x
                CoreCoord a_sender_device = mesh_device->worker_core_from_logical_core({0, 0});
                CoreCoord b_recv_start_device = mesh_device->worker_core_from_logical_core({x, 1});
                CoreCoord b_recv_end_device = mesh_device->worker_core_from_logical_core({x, core_grid.y - 1});

                SetRuntimeArgs(program, reader_top_row_id, core_logical,
                    {src1_buf->address(), Kt, Nt, block_col_start,
                     M_block_tiles, N_block_tiles, K_block_tiles, num_k_blocks,
                     // A receive
                     static_cast<uint32_t>(a_sender_device.x),
                     static_cast<uint32_t>(a_sender_device.y),
                     receivers_ready_A, slab_sent_A,
                     // B multicast
                     static_cast<uint32_t>(b_recv_start_device.x),
                     static_cast<uint32_t>(b_recv_start_device.y),
                     static_cast<uint32_t>(b_recv_end_device.x),
                     static_cast<uint32_t>(b_recv_end_device.y),
                     receivers_ready_B, slab_sent_B, num_b_receivers});

            } else {
                // Interior core: receives A from (0, y), receives B from (x, 0)
                CoreCoord a_sender_device = mesh_device->worker_core_from_logical_core({0, y});
                CoreCoord b_sender_device = mesh_device->worker_core_from_logical_core({x, 0});

                SetRuntimeArgs(program, reader_interior_id, core_logical,
                    {M_block_tiles, N_block_tiles, K_block_tiles, num_k_blocks,
                     // A receive
                     static_cast<uint32_t>(a_sender_device.x),
                     static_cast<uint32_t>(a_sender_device.y),
                     receivers_ready_A, slab_sent_A,
                     // B receive
                     static_cast<uint32_t>(b_sender_device.x),
                     static_cast<uint32_t>(b_sender_device.y),
                     receivers_ready_B, slab_sent_B});
            }
        }
    }

    // ---- Execute ----
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return dst_tensor.to_vector<bfloat16>();
}

int main(int argc, char** argv) {
    try {
        uint32_t grid_x = 5;
        uint32_t grid_y = 5;
        uint32_t K_block = 2;

        if (argc >= 3) {
            grid_x = static_cast<uint32_t>(std::stoi(argv[1]));
            grid_y = static_cast<uint32_t>(std::stoi(argv[2]));
        }
        if (argc >= 4) {
            K_block = static_cast<uint32_t>(std::stoi(argv[3]));
        }

        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr uint32_t M = 640;
        constexpr uint32_t K = 320;
        constexpr uint32_t N = 640;

        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<bfloat16> a(M * K), b(K * N);
        for (auto& v : a) v = bfloat16(dist(rng));
        for (auto& v : b) v = bfloat16(dist(rng));

        CoreCoord core_grid{grid_x, grid_y};
        std::cout << "Running Lab3 Ex4 (Multicast MatMul) on " << grid_x << "x" << grid_y << " grid...\n";

        auto result = matmul_multicast(a, b, M, K, N, mesh_device, core_grid, K_block);

        auto ref = reference_matmul(a, b, M, K, N);
        verify_against_reference(ref, result);

        mesh_device->close();
        std::cout << "Test Passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
