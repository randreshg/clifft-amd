#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace clifft {
struct CompiledModule;
}  // namespace clifft

namespace clifft_cuda {

struct SurvivorCounts {
    uint64_t total_shots = 0;
    uint64_t passed_shots = 0;
    uint64_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;
    double kernel_seconds = 0.0;
};

struct CudaSamplerOptions {
    uint64_t seed = 0;
    uint32_t block_size = 256;
    bool keep_records = false;
};

// Native sampler target:
// - The caller hands over the exact Clifft CompiledModule, including bytecode
//   and ConstantPool.
// - The implementation flattens that module to GPU POD buffers.
// - The GPU returns aggregate survivor/error counters by default.
SurvivorCounts sample_survivors_cuda(const clifft::CompiledModule& program, uint64_t shots,
                                     const CudaSamplerOptions& options);

std::string cuda_backend_diagnostics();

}  // namespace clifft_cuda
