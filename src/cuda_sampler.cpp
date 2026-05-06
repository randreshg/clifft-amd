#include "clifft_cuda/cuda_sampler.h"

#include <stdexcept>

namespace clifft_cuda {

SurvivorCounts sample_survivors_cuda(const clifft::CompiledModule&, uint64_t,
                                     const CudaSamplerOptions&) {
    throw std::runtime_error(
        "clifft-cuda native sampler is not built in this workspace. "
        "A CUDA toolkit (nvcc/cuda.h or NVRTC/cuda.h) is required.");
}

std::string cuda_backend_diagnostics() {
    return "native CUDA sampler stub: CUDA toolkit is not available in this workspace";
}

}  // namespace clifft_cuda
