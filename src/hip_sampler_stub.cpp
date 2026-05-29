#include "clifft_cuda/hip_sampler.h"

#include <stdexcept>

namespace clifft_cuda {

SurvivorCounts sample_survivors_hip(const clifft::CompiledModule&, uint64_t,
                                    const HipSamplerOptions&) {
    throw std::runtime_error(
        "clifft-amd native HIP sampler is not built in this workspace. "
        "Reconfigure with -DCLIFFT_AMD_ENABLE_HIP=ON and a ROCm/HIP toolchain "
        "(hipcc, libamdhip64, hip/hip_runtime.h) targeting gfx942.");
}

std::string hip_backend_diagnostics() {
    return "native HIP sampler stub: HIP backend not built "
           "(ROCm/HIP toolchain is not available in this workspace)";
}

}  // namespace clifft_cuda
