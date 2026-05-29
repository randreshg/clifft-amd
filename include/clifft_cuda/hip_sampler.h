#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The HIP backend mirrors the CUDA backend exactly. To avoid duplicating the
// option/struct shapes (and the resulting symbol/definition divergence), reuse
// the existing SurvivorCounts and CudaSamplerOptions from the CUDA header. Both
// backends live in namespace clifft_cuda by design (see README.md
// §4): the namespace name is historical and is shared, not CUDA-exclusive.
#include "clifft_cuda/cuda_sampler.h"

namespace clifft {
struct CompiledModule;
}  // namespace clifft

namespace clifft_cuda {

// The HIP sampler takes the same options as the CUDA sampler. Expose a HIP-named
// alias so callers can name the type after their backend without introducing a
// second, drift-prone definition.
using HipSamplerOptions = CudaSamplerOptions;

// Native HIP sampler target (AMD MI300X / gfx942 / wavefront 64):
// - The caller hands over the exact Clifft CompiledModule, including bytecode
//   and ConstantPool.
// - The implementation flattens that module to GPU POD buffers.
// - The GPU returns aggregate survivor/error counters by default.
//
// This backend reproduces the CUDA backend's RNG/seed semantics and integer
// survivor accounting. Single-precision amplitude math is compiled with
// -ffp-contract=off so it rounds reproducibly; parity is enforced by the
// same-seed determinism check and the CPU-oracle statistical cross-check in
// README.md (Verification) (not asserted as cross-vendor bit-for-bit).
SurvivorCounts sample_survivors_hip(const clifft::CompiledModule& program, uint64_t shots,
                                    const HipSamplerOptions& options);

std::string hip_backend_diagnostics();

}  // namespace clifft_cuda
