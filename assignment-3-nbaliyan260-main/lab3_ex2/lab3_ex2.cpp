// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lab 3, Exercise 2: Extended Multicast Example
// The sender core now also participates in compute and writeback.
// Output: 4 copies of the input tensor (1 sender + 3 receivers).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

using namespace std;
using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace ttnn;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

struct ProgramState {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    Program program;
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range;
    tt::tt_metal::distributed::MeshCommandQueue& cq;

    ProgramState(
        std::shared_ptr<distributed::MeshDevice> mesh_device,
        Program program,
        tt::tt_metal::distributed::MeshWorkload workload,
        tt::tt_metal::distributed::MeshCoordinateRange device_range,
        tt::tt_metal::distributed::MeshCommandQueue& cq) :
        mesh_device(std::move(mesh_device)),
        program(std::move(program)),
        workload(std::move(workload)),
        device_range(std::move(device_range)),
        cq(cq) {}
};

ProgramState init_program() {
    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    return ProgramState(std::move(mesh_device), std::move(program), std::move(workload), std::move(device_range), cq);
}

void create_cb(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& cores,
    uint32_t num_tiles,
    tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
        .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, cores, cb_config);
}

bool verify_multicast_results(
    const std::vector<bfloat16>& reference,
    const std::vector<bfloat16>& received,
    uint32_t n_tiles,
    uint32_t num_copies) {
    log_info(tt::LogAlways, "=========== MULTICAST TENSOR VERIFICATION ===========");

    const uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    const uint32_t total_elements = n_tiles * elements_per_tile;

    TT_FATAL(reference.size() == total_elements, "Reference size mismatch");
    TT_FATAL(received.size() == num_copies * total_elements, "Received size mismatch");

    bool all_pass = true;

    for (uint32_t copy = 0; copy < num_copies; copy++) {
        uint32_t mismatch_count = 0;
        uint32_t first_mismatch_idx = 0;

        for (uint32_t i = 0; i < total_elements; i++) {
            uint32_t received_idx = (copy * total_elements) + i;
            if (received[received_idx] != reference[i]) {
                if (mismatch_count == 0) {
                    first_mismatch_idx = i;
                }
                mismatch_count++;
            }
        }

        if (mismatch_count == 0) {
            log_info(tt::LogAlways, "[PASS] Core {} produced correct output ({} tiles)", copy, n_tiles);
        } else {
            log_error(
                tt::LogAlways,
                "[FAIL] Core {} has {} mismatches (first at index {})",
                copy,
                mismatch_count,
                first_mismatch_idx);
            all_pass = false;
        }
    }

    if (all_pass) {
        log_info(tt::LogAlways, "[PASS] All {} cores produced correct output", num_copies);
    } else {
        log_error(tt::LogAlways, "[FAIL] One or more cores have incorrect data");
    }

    log_info(tt::LogAlways, "=====================================================");

    return all_pass;
}

void multicast_tensor_tensix(
    const std::vector<bfloat16>& input_data,
    std::vector<bfloat16>& output_data,
    const uint32_t M,
    const uint32_t N,
    const uint32_t num_receivers,
    ProgramState& prog_state) {
    const uint32_t total_elements = M * N;
    constexpr uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    const uint32_t num_total_cores = num_receivers + 1;  // sender + receivers

    TT_FATAL(input_data.size() == total_elements, "Input data size must be M * N");
    TT_FATAL(output_data.size() == num_total_cores * total_elements,
        "Output data size must be num_total_cores * M * N");

    TT_FATAL(total_elements % elements_per_tile == 0, "Total elements must be divisible by elements per tile");
    const uint32_t n_tiles = total_elements / elements_per_tile;

    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));

    TensorSpec input_spec(Shape({M, N}), tile_layout);
    TensorSpec output_spec(Shape({num_total_cores * M, N}), tile_layout);

    Tensor src_tensor = Tensor::from_vector<bfloat16>(input_data, input_spec, prog_state.mesh_device.get());
    Tensor dst_tensor = create_device_tensor(output_spec, prog_state.mesh_device.get());

    auto src_mesh_buffer = src_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    // Core ranges
    CoreRange all_cores_logical = CoreRange({0, 0}, {num_receivers, 0});
    CoreCoord sender_core_logical = {0, 0};
    CoreRange receiver_cores_logical = CoreRange({1, 0}, {num_receivers, 0});

    // Device coordinates for NoC multicast
    CoreCoord sender_core_device = prog_state.mesh_device->worker_core_from_logical_core(sender_core_logical);
    CoreRange receiver_cores_device = CoreRange(
        prog_state.mesh_device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
        prog_state.mesh_device->worker_core_from_logical_core(receiver_cores_logical.end_coord));

    uint32_t num_dests = static_cast<uint32_t>(receiver_cores_logical.size());
    TT_FATAL(num_dests == num_receivers, "Number of receiver cores must match num_receivers parameter");

    // Semaphores
    uint32_t receivers_ready_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t tile_sent_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, INVALID);

    // Circular buffers: on ALL cores (sender + receivers)
    constexpr uint32_t tiles_per_cb = 2;
    create_cb(prog_state.program, all_cores_logical, tiles_per_cb, tt::CBIndex::c_0);
    create_cb(prog_state.program, all_cores_logical, tiles_per_cb, tt::CBIndex::c_16);

    // Sender reader kernel (reads from DRAM + multicasts)
    std::vector<uint32_t> mcast_sender_compile_args;
    TensorAccessorArgs(*src_mesh_buffer).append_to(mcast_sender_compile_args);
    DataMovementConfig mcast_sender_config = {
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = mcast_sender_compile_args};

    KernelHandle mcast_sender_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex2/kernels/dataflow/mcast_sender.cpp",
        sender_core_logical,
        mcast_sender_config);

    // Receiver reader kernel
    DataMovementConfig mcast_receiver_config = {
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};

    KernelHandle mcast_receiver_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex2/kernels/dataflow/mcast_receiver.cpp",
        receiver_cores_logical,
        mcast_receiver_config);

    // Writer kernel on ALL cores (sender + receivers)
    std::vector<uint32_t> write_tiles_compile_args;
    TensorAccessorArgs(*dst_mesh_buffer).append_to(write_tiles_compile_args);
    DataMovementConfig write_tiles_config = {
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = write_tiles_compile_args};

    KernelHandle write_tiles_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex2/kernels/dataflow/write_tiles.cpp",
        all_cores_logical,
        write_tiles_config);

    // Compute kernel on ALL cores (sender + receivers)
    vector<uint32_t> tiles_copy_compile_args = {n_tiles};
    CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_ex2/kernels/compute/tiles_copy.cpp",
        all_cores_logical,
        tt_metal::ComputeConfig{.compile_args = tiles_copy_compile_args});

    // Runtime args for sender
    SetRuntimeArgs(
        prog_state.program,
        mcast_sender_id,
        sender_core_logical,
        {static_cast<uint32_t>(receiver_cores_device.start_coord.x),
         static_cast<uint32_t>(receiver_cores_device.start_coord.y),
         static_cast<uint32_t>(receiver_cores_device.end_coord.x),
         static_cast<uint32_t>(receiver_cores_device.end_coord.y),
         receivers_ready_semaphore,
         tile_sent_semaphore,
         src_mesh_buffer->address(),
         n_tiles,
         num_dests});

    // Runtime args for receivers
    SetRuntimeArgs(
        prog_state.program,
        mcast_receiver_id,
        receiver_cores_logical,
        {static_cast<uint32_t>(sender_core_device.x),
         static_cast<uint32_t>(sender_core_device.y),
         receivers_ready_semaphore,
         tile_sent_semaphore,
         n_tiles});

    // Writer args: sender core writes at index 0, receivers at 1..num_receivers
    SetRuntimeArgs(
        prog_state.program,
        write_tiles_id,
        sender_core_logical,
        {dst_mesh_buffer->address(), n_tiles, static_cast<uint32_t>(0)});

    int receiver_idx = 1;
    for (const CoreCoord& core : receiver_cores_logical) {
        SetRuntimeArgs(
            prog_state.program,
            write_tiles_id,
            core,
            {dst_mesh_buffer->address(), n_tiles, static_cast<uint32_t>(receiver_idx)});
        receiver_idx++;
    }

    log_info(tt::LogAlways, "Launching multicast of {} tiles: sender + {} receivers = {} total cores",
        n_tiles, num_receivers, num_total_cores);

    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    tt_metal::distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);

    log_info(tt::LogAlways, "Multicast complete");

    output_data = dst_tensor.to_vector<bfloat16>();
}

int main() {
    bool pass = true;

    try {
        constexpr uint32_t num_receivers = 3;
        constexpr uint32_t num_total_cores = num_receivers + 1;
        constexpr uint32_t M = 640;
        constexpr uint32_t N = 640;
        constexpr uint32_t total_elements = M * N;
        constexpr uint32_t n_tiles = total_elements / (TILE_HEIGHT * TILE_WIDTH);

        log_info(
            tt::LogAlways,
            "Lab3 Ex2: {}x{} tensor ({} tiles), sender + {} receivers = {} total cores",
            M, N, n_tiles, num_receivers, num_total_cores);

        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<float> rng_dist(0.f, 1.0f);

        std::vector<bfloat16> input_data(total_elements);
        for (bfloat16& v : input_data) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        std::vector<bfloat16> output_data(num_total_cores * total_elements);

        ProgramState prog_state = init_program();

        multicast_tensor_tensix(input_data, output_data, M, N, num_receivers, prog_state);

        log_info(tt::LogAlways, "Output vector size: {} elements", output_data.size());

        pass = verify_multicast_results(input_data, output_data, n_tiles, num_total_cores);

        pass &= prog_state.mesh_device->close();

    } catch (const std::exception& e) {
        log_error(tt::LogAlways, "Test failed with exception!");
        log_error(tt::LogAlways, "{}", e.what());
        throw;
    }

    if (pass) {
        log_info(tt::LogAlways, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
