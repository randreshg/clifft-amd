# clifft-cuda

`clifft-cuda` runs [Clifft](https://github.com/unitaryfoundation/clifft)-compiled Stim circuits with a native CUDA sampler.

This fork (`clifft-amd`) adds a native **AMD ROCm/HIP** backend beside the CUDA
one, so the same sampler runs on AMD Instinct GPUs. It is verified on MI300X
(`gfx942`, ROCm 7.2.0). The CUDA path is unchanged and the HIP backend is opt-in:
see [Running on AMD GPUs](#running-on-amd-gpus-rocm--hip) below and the HIP build
section in [instruction.md](instruction.md).

The CUDA backend is implemented with the help of Codex 5.5.

For build/setup instructions, see [instruction.md](instruction.md).

## Performance

for distance-5 magic state cultivation circuits with noise level p=0.1%:

| Tool | processor | power | sampling speed | speed up |
|-----|-----|-----|-----|-----|
|Clifft|Xeon(R) Gold 5218R CPU @ 2.10GHz|125 W|~59,800 shots/s|-|
|Clifft-cuda|NVIDIA RTX PRO 5000|300 W|~1,180,000 shots/s|19.7|
|Clifft-amd (this fork)|AMD Instinct MI300X|750 W|~1,200,000 shots/s|~20|

We sampled 100 billion shots in 24 hours and obtained an after-postselection logical error rate of 3.54e-09 (with 51 errors found).

## Run A Circuit

From the `clifft-cuda` directory:

```bash
mkdir -p runs

./build-core-cuda/run_msc_cuda \
  --circuit ./circuit_d5_p=0.001.stim \
  --shots 10000000 \
  --seed 42 \
  --block-size 256 \
  --postselection all \
  > runs/d5_p0.001_shots10000000_seed42_cuda.stdout.json \
  2> runs/d5_p0.001_shots10000000_seed42_cuda.stderr.log
```

Use `--postselection none` to keep every shot and run the full circuit without
detector-based discard.

## Running on AMD GPUs (ROCm / HIP)

Build with the HIP option (see [instruction.md](instruction.md) for details),
then run `run_msc_hip` with the same flags as `run_msc_cuda`:

```bash
cmake -S . -B build-core-hip \
  -DCLIFFT_AMD_ENABLE_HIP=ON \
  -DCLIFFT_SOURCE_DIR=./third_party/clifft \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-core-hip -j

./build-core-hip/run_msc_hip --diagnose          # lists the AMD GPUs
./build-core-hip/run_msc_hip \
  --circuit ./circuit_d5_p=0.001.stim --shots 10000000 --seed 42 \
  --block-size 256 --postselection all
```

### What the port changes

The HIP backend lives beside the CUDA one and reuses everything else unchanged:

- `src/hip_sampler.hip` is the HIP version of `src/cuda_sampler.cu`. It is a
  mechanical transform: the include becomes `<hip/hip_runtime.h>`, the CUDA
  runtime calls become their HIP equivalents (`cudaMalloc` to `hipMalloc`,
  `cudaMemcpy` to `hipMemcpy`, the event and device-property calls likewise), and
  the diagnostics print the AMD `gcnArchName`. No kernel, RNG, reduction, atomic,
  or dispatch logic changes.
- `tools/run_msc_hip.cpp` is the runner, with flags and JSON identical to
  `run_msc_cuda`.
- `CMakeLists.txt` gains a default-OFF `CLIFFT_AMD_ENABLE_HIP` option that builds
  the HIP library and `run_msc_hip`. A CUDA-only, HIP-only, or both-off configure
  each behaves exactly as before.
- Two AMD-specific build details: the host runner links `hip::host` (not
  `hip::device`, which leaks `--offload-arch` into the host compiler), and the HIP
  device library is built with `-ffp-contract=off` so single-precision amplitude
  math rounds reproducibly.

**Wavefront size.** CUDA assumes 32-lane warps; MI300X uses 64-lane wavefronts.
The sampler has no warp-level code (no shuffles, ballots, or lane masks), and
every reduction is a full-block `__syncthreads`, which is correct at any
wavefront width. The port therefore needs no wave64-specific changes.

### Verification

The same binary hosts the CPU reference. `run_msc_hip --cpu-reference` runs the
trusted `clifft` CPU sampler on the identical compiled circuit. CPU and HIP do
not match shot for shot, because each uses independent per-shot RNG substreams,
so they are compared on aggregate rates. `scripts/compare_oracle.py` runs both
and checks each rate within a 5-sigma band:

```bash
scripts/compare_oracle.py --circuit ./circuit_d5_p=0.001.stim \
  --shots 1000000 --seed 42 --postselection all
```

On MI300X (`gfx942`, ROCm 7.2.0) the HIP backend matches the CPU reference within
5 sigma across all three kernel regimes, it is deterministic for a fixed seed,
and it reaches about 1.2 M shots/s on the d5 circuit:

| Kernel (selected by `peak_rank`) | circuit | check | CPU | HIP |
|---|---|---|---|---|
| per-thread (`<=4`) | pure-Clifford | logical-error rate | 0.499590 | 0.499707 |
| shared-block (`5..10`) | d5, p=0.001 | survival rate | 0.143885 | 0.143519 |
| global-block (`11..19`) | d7, p=0.0005 | logical-error rate | 0.500200 | 0.512400 |

## Output

Results are written wherever stdout/stderr are redirected. The recommended
location is `runs/`:

- `runs/*.stdout.json`: final aggregate sampling result.
- `runs/*.stderr.log`: progress messages and runtime errors.

The JSON includes fields such as `shots`, `seed`, `peak_rank`, `passed_shots`,
`discarded_shots`, `logical_errors`, `discard_rate`, `survival_rate`,
`kernel_seconds`, and throughput.

## Quick GPU Check

```bash
./build-core-cuda/run_msc_cuda --diagnose    # NVIDIA
./build-core-hip/run_msc_hip --diagnose      # AMD
```
